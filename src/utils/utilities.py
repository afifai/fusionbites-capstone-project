import matplotlib.pyplot as plt
import os
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from src.utils.data_preparation import split_path

def create_results_folder():
    """
    Creates required directories for storing results if they do not exist. 
    
    This function specifically creates the following directories:
    - 'results/images': Directory to store images related to the project
    - 'results/models': Directory to store model files
    - 'results/images/plots': Directory to store plots created during the analysis
    - 'results/images/fusionbites_candidates': Directory to store images classified as potential 'FusionBites'

    Returns
    -------
    None
    """
    paths = [
        'results/images',
        'results/models',
        'results/images/plots',
        'results/images/fusionbites_candidates'
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def define_callback(path):
    """
    Defines a callback function for model training.

    Parameters:
    - path (str): The path where the model will be saved.

    Returns:
    - ModelCheckpoint object: The callback for model training.
    """
    model_checkpoint_callback = ModelCheckpoint(
        filepath=path,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    return model_checkpoint_callback


def plot_curves(history, model_name):
    """
    Plots the loss and accuracy curves for the model.

    Parameters:
    - history (History object): The history object obtained from the model training.
    - model_name (str): The name of the model.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title(f'Loss {model_name}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"results/images/plots/{model_name} Loss.png")

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title(f'Accuracy {model_name}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"results/images/plots/{model_name} Accuracy.png")


def move_candidate(pred_result, filenames, thresh=0.5):
    """
    Moves the candidate images to a specific folder.

    Parameters:
    - pred_result (list): The predicted results.
    - filenames (list): The list of filenames.
    - thresh (float): The threshold value for the prediction.
    """
    print("[INFO] MOVING CANDIDATE ...")
    source_path = 'data/test/'
    dest_path = 'results/images/fusionbites_candidates/'

    for idx, item in enumerate(pred_result):
        if item[0] > thresh and item[1] > thresh:
            alternate_filename = '_'.join(split_path(filenames[idx]))
            shutil.copy(os.path.join(source_path, filenames[idx]), os.path.join(dest_path, alternate_filename))


def predict_data(model, test_data, test_size):
    """
    Predicts the test dataset using the trained model.

    Parameters:
    - model (Model object): The trained model.
    - test_data (ImageDataGenerator object): The test data.
    - test_size (int): The size of the test data.

    Returns:
    - numpy array: The predicted results.
    """
    test_data.batch_size = test_size
    data, label = test_data.next()
    return model.predict(data)
