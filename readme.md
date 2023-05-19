# Capstone Project : FusionBites

## Project Overview

The project aims to automate the identification of images that align with the concept of "FusionBites", a new, unique food concept that blends sushi and sandwiches. This would involve analyzing a dataset composed of images tagged as either sushi or sandwiches.

## Problem Statement

The challenge lies in creating an automated workflow or method capable of effectively and accurately identifying potential instances of "FusionBites" based on the given dataset. This task includes parsing through images labeled as sushi or sandwiches and determining whether they could potentially be classified as "FusionBites", a combination of the two.

## Methodology

This project uses a Convolutional Neural Network (CNN) based approach to perform multilabel classification of images. Two types of architectures were experimented with:

1. **Custom CNN**: A simple custom-built CNN model with multiple Conv2D layers followed by MaxPooling2D layers. This model also incorporates Dropout layers for regularization.

2. **MobileNetV2**: A pre-trained MobileNetV2 model has been used as a base model with an additional Dense layer at the top to perform the multilabel classification.

The models were trained using the binary cross-entropy loss function, which is suitable for multilabel classification problems.

Images were preprocessed and augmented using `ImageDataGenerator` from Keras, which can generate batches of tensor image data with real-time data augmentation.

The data was split into training and testing sets, and the models were trained on the training set while validation was performed on the testing set.

## Metrics

The performance of the models were evaluated using the following metrics:

1. **Accuracy**: This is the proportion of the total number of predictions that were correct. It is a useful measure when the target classes are balanced.

2. **Loss (Binary Cross Entropy)**: Since this is a multilabel classification problem, binary cross-entropy loss was used. It measures the performance of a classification model whose output is a probability value between 0 and 1. The loss increases as the predicted probability diverges from the actual label.

The training process records these metrics for both training and validation data for each epoch, allowing for evaluation of how well the model is learning over time.

Note: Given the multilabel nature of the task, accuracy might not be the best metric. It might be beneficial to consider additional metrics like Precision, Recall, F1-score, or use a multi-label confusion matrix for a more detailed performance analysis.

## Prediction Criteria

Post-training, the trained models are used to predict on the test set. For each prediction, the model outputs probabilities for each class. If the probability of a class is greater than a specified threshold, the image is considered as belonging to that class. Specifically for this project, an image is considered a "FusionBites" food item if the model's predicted probability is above the specified threshold for that class.

These predictions can be used to classify new images into their respective categories, enabling an automated categorization process for food images. The results of the predictions, including classified images and performance metrics, are saved in the `results` directory for further analysis and review.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```shell
python 3.9+
pip
```

### Installing

Clone the repository:

```shell
git clone https://github.com/yourusername/yourrepository.git
```

Install dependencies:

```shell
pip install -r requirements.txt
```

### Configuration

Configuration for the model training and evaluation can be done via the `config.json` file.
Here's a brief explanation of each item:

1. `"download_link"`: This is the link to download the dataset. The model will use the images in this zip file for training and testing.

2. `"train_dir"` and `"test_dir"`: These are the directories where train and test data respectively are stored after being downloaded and unzipped.

3. `"model_output"`: This is the location where trained model will be saved.

4. `"model"`: This is the selection of what model that we used for training (1 for CustomCNN and 2 for Fine Tuning MobileNetV2, please refer to `src.models.neural_net.py` file).

5. `"test_size"`: This is represent of the number of test images used in model evaluation.

6. `"epochs"`: This is the number of times the learning algorithm will work through the entire training dataset.

7. `"batch_size"`: This is the number of training examples utilized in one iteration.

8. `"thresh"`: This is the threshold for the output neuron activation, which determines whether a particular label (sushi or sandwich or both) should be activated. For example, in this case, any output value above 0.35 will be considered as an active label.

9. `"seed"`: This is the seed for the random number generator. It is used to ensure that your experiments can be reproduced exactly.

10. `"learning_rate"`: This is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

### Data

Data will downloaded automatically when we first time run the training script.

### Training

To train the model, run:

```shell
python run_train.py
```

## Built With

- [Tensorflow](https://www.tensorflow.org/) - The deep learning framework used.
- [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) - The convolutional neural network architecture used.

## Medium Post

Read more about this project and the concept behind it in this [blog post](https://medium.com/@afifakbariskandar_42661/fusionbites-unleashing-the-power-of-ai-for-culinary-innovation-55b9bbfe8ad8) on Medium
