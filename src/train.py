import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
from sklearn.metrics import classification_report
from src.utils.data import get_data_and_split, config_data_generator
from src.models.neural_net import CustomCNN, MobileNetCNN
from src.utils.data_preparation import load_config
from src.utils.utilities import create_results_folder, define_callback, plot_curves, predict_data, move_candidate


def main():
    """
    Main function for executing the fusionbites project.
    """
    # Create sub-folder in results
    create_results_folder()

    # Load the configuration from the JSON file
    config = load_config('config.json')

    # Set the seed for TensorFlow's random number generator
    set_seed(config['seed'])

    # Download and split the data
    get_data_and_split(link=config['download_link'],
                       test_size=config['test_size'],
                       seed=config['seed'])

    # Generate the train and test data
    train_data, test_data = config_data_generator(
                                train_dir=config['train_dir'],
                                test_dir=config['test_dir'],
                                batch_size=config['batch_size'],
                                seed=config['seed'])

    # Define the available models
    models = {
        1: [CustomCNN, 'Custom CNN Model'],
        2: [MobileNetCNN, 'Mobile Net V2 Model']
    }

    print("[INFO] COMPILING MODEL ...")
    opt = Adam(learning_rate=config['learning_rate'])
    model = models[config['model']][0].build(
                                          width=224,
                                          height=224,
                                          depth=3,
                                          classes=2)
    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] TRAINING NETWORK ...")
    callback = define_callback(path=config['model_output'])
    H = model.fit(train_data,
                  epochs=config['epochs'],
                  validation_data=test_data,
                  callbacks=[callback])

    # Plot the training curves
    plot_curves(H, models[config['model']][1])

    # Load the best model and predict the test data
    best_model = load_model(config['model_output'])

    # calculate the accuracy of the model
    _, accuracy = best_model.evaluate_generator(test_data, steps=len(test_data), verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))
    # generate prediction
    Y_pred = model.predict_generator(test_data, steps=len(test_data), verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)

    # generate classification report
    print(classification_report(test_data.classes, y_pred, target_names=test_data.class_indices))

    result = predict_data(best_model, test_data)

    # Move the candidate images to the specified folder
    move_candidate(result, test_data.filenames, thresh=config['thresh'])
