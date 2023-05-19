"""
This module contains functions to download and configure the data for training and testing.
"""

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.random import set_seed

from src.utils.data_preparation import download_data, extract_data, split_data


def get_data_and_split(link, test_size, seed):
    """
    Download, extract, and split the dataset into train and test sets if not already exists.

    Parameters:
    - link (str): The URL to download the dataset.
    - test_size (float): The proportion of the dataset to include in the test split.
    - seed (int): Random seed for reproducibility.
    """
    if os.path.exists('data/train'):
        print("[INFO] DATA ALREADY EXISTS")
        return

    if not os.path.exists('data/sushi_or_sandwich_photos.zip'):
        download_data(link)

    extract_data()
    split_data(test_size, seed)


def config_data_generator(train_dir, test_dir, val_split=0.1, batch_size=128, seed=42):
    """
    Configures the data generators for the training and testing sets.

    Parameters:
    - train_dir (str): Directory with training set images.
    - test_dir (str): Directory with test set images.
    - val_split (float, optional): The proportion of the training set to include in the validation split.
    - batch_size (int, optional): Number of samples per gradient update.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - train_data: Data generator for the training set.
    - test_data: Data generator for the test set.
    """
    set_seed(seed)
    train_datagen = ImageDataGenerator(validation_split=val_split,
                                       rescale=1./255,
                                       rotation_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(train_dir,
                                                   batch_size=batch_size,
                                                   target_size=(224, 224),
                                                   class_mode="categorical",
                                                   seed=seed,
                                                   shuffle=True)

    test_data = test_datagen.flow_from_directory(test_dir,
                                                 batch_size=batch_size,
                                                 target_size=(224, 224),
                                                 class_mode="categorical",
                                                 seed=seed,
                                                 shuffle=False)
    return train_data, test_data
