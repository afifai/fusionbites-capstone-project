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

Post-training, the trained models are used to predict on the test set. For each prediction, the model outputs probabilities for each class. If the probability of a class is greater than a specified threshold (typically 0.5, but we can change it in `config.json`), the image is considered as belonging to that class. Specifically for this project, an image is considered a "FusionBites" food item if the model's predicted probability is above the specified threshold for that class.

These predictions can be used to classify new images into their respective categories, enabling an automated categorization process for food images. The results of the predictions, including classified images and performance metrics, are saved in the `results` directory for further analysis and review.

## Analysis and Results

TODO
