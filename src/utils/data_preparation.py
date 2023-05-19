import os
import json
import requests
import glob
import re
import random
from tqdm import tqdm
from zipfile import ZipFile
import shutil

def load_config(config_file='config.json'):
    """
    Loads configuration data from a JSON file.
    
    Parameters:
    - config_file (str): The path to the JSON configuration file. Defaults to 'config.json'.

    Returns:
    - data (dict): The configuration data.
    """
    with open(config_file, "r") as jsonfile:
        data = json.load(jsonfile)
    return data


def create_train_test_folder():
    """
    Creates the directories for storing train and test data if they do not already exist.
    """
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)


def download_data(link, test_size):
    """
    Downloads the dataset and displays a progress bar.

    Parameters:
    - link (str): The URL to download the dataset from.
    - test_size (int): The size of the test dataset.
    """
    create_train_test_folder()
    print("[INFO] DOWNLOADING DATASET ...")

    response = requests.get(link, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open('data/sushi_or_sandwich_photos.zip', 'wb') as file, tqdm(
            desc='sushi_or_sandwich_photos.zip',
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            bar.update(size)


def extract_data():
    """
    Extracts dataset from the zip file.
    """
    print("[INFO] EXTRACTING DATASET ...")
    with ZipFile('data/sushi_or_sandwich_photos.zip', 'r') as zip_file:
        zip_file.extractall('data')


def split_data(test_size, seed):
    """
    Splits the data into train and test sets.

    Parameters:
    - test_size (int): The size of the test dataset.
    - seed (int): The seed for the random number generator.
    """
    print("[INFO] SPLITTING TRAIN TEST DATASET ...")
    train_test_split(test_size, seed)
    shutil.rmtree('data/sushi_or_sandwich/')


def split_path(path):
    """
    Splits a file path into its constituent parts.

    Parameters:
    - path (str): The path to be split.

    Returns:
    - list: A list of the split parts of the path.
    """
    return re.split('/|\\\\', path)


def move_data(path_data_list, parent, dest):
    """
    Moves data from one directory to another.

    Parameters:
    - path_data_list (list): A list of file paths to move.
    - parent (str): The current directory of the files.
    - dest (str): The destination directory for the files.
    """
    for item in path_data_list:
        filename = split_path(item)[-1]
        os.rename(os.path.join(parent, filename), os.path.join(dest, filename))


def train_test_split(test_size, seed):
    """
    Splits image data into training and testing datasets.

    Parameters:
    - test_size (int): The size of the test dataset.
    - seed (int): The seed for the random number generator.
    """
    random.seed(seed)
    for path in glob.glob("data/sushi_or_sandwich/*/"):
        path_split = split_path(path)
        os.makedirs(f'data/train/{path_split[-2]}', exist_ok=True)
        os.makedirs(f'data/test/{path_split[-2]}', exist_ok=True)

        images_train = glob.glob(path + '*.jpg')
        random.shuffle(images_train)

        images_test = images_train[-(test_size//2):]
        images_train = images_train[:-(test_size//2)]

        move_data(images_train, path, os.path.join(path_split[0], 'train', path_split[2]))
        move_data(images_test, path, os.path.join(path_split[0], 'test', path_split[2]))
