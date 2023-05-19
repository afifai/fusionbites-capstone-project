"""
This module contains classes to build a Custom CNN and a MobileNetV2 CNN for image classification.
"""

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D


class CustomCNN:
    """
    This class defines a custom Convolutional Neural Network (CNN) for image classification.
    """
    @staticmethod
    def build(width, height, depth, classes):
        """
        Builds and returns the custom CNN model.
        
        Parameters:
        - width (int): The width of the input images.
        - height (int): The height of the input images.
        - depth (int): The number of channels of the input images.
        - classes (int): The number of classes for the classification.
        
        Returns:
        - model: The custom CNN model.
        """
        model = Sequential()
        input_shape = (height, width, depth)

        model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model


class MobileNetCNN:
    """
    This class defines a Convolutional Neural Network (CNN) using MobileNetV2 for image classification.
    """
    @staticmethod
    def build(width, height, depth, classes):
        """
        Builds and returns the MobileNetV2 CNN model.
        
        Parameters:
        - width (int): The width of the input images.
        - height (int): The height of the input images.
        - depth (int): The number of channels of the input images.
        - classes (int): The number of classes for the classification.
        
        Returns:
        - model: The MobileNetV2 CNN model.
        """
        input_shape = (height, width, depth)
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False

        inputs = Input(shape=input_shape)
        model = base_model(inputs, training=False)
        model = GlobalAveragePooling2D()(model)
        model = Dropout(0.2)(model)
        model = Dense(classes)(model)
        outputs = Activation('sigmoid')(model)
        model = Model(inputs, outputs)

        return model
