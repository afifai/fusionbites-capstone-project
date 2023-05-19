from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
from data import get_data_and_split, config_data_generator
from neural_net import CustomCNN, MobileNetCNN
from util import define_callback, plot_curves, predict_data, move_candidate, load_config

def main():
    """
    Main function for executing the fusionbites project.
    """
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
                  steps_per_epoch=len(train_data),
                  validation_data=test_data,
                  validation_steps=len(test_data),
                  callbacks=[callback])

    # Plot the training curves
    plot_curves(H, models[config['model']][1])

    # Load the best model and predict the test data
    best_model = load_model(config['model_output'])
    result = predict_data(best_model, test_data, test_size=config['test_size'])

    # Move the candidate images to the specified folder
    move_candidate(result, test_data.filenames)


if __name__ == '__main__':
    main()
