import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

EPOCHS = 1
IMG_WIDTH = 150
IMG_HEIGHT = 150
NUM_CATEGORIES = 6
TEST_SIZE = 0.4
BATCH_SIZE = 32


def main():
    # Load image data from directory
    data_dir = "./datasets-v2"

    x_train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    y_test = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    # Optimize dataset for performance
    x_train = x_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    y_test = y_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Get a compiled neural network
    model = get_model()
    model.summary()
    # # Fit model on training data
    model.fit(x_train, validation_data=y_test, epochs=EPOCHS)

    # # Evaluate neural network performance
    model.evaluate(y_test, verbose=2)

    # # Save model to file
    filename = "dental_model.keras"
    model.save(filename)
    print(f"Model saved to {filename}.")


def get_model() -> Model:
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Data augmentation using the following Keras preprocessing layers
    x = layers.RandomFlip('horizontal')(img_input)
    x = layers.RandomRotation(0.1)(img_input)
    x = layers.RandomZoom(0.1)(x)

    # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
    x = layers.Rescaling(1./255)(x)

    # First convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Convolution2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor
    x = layers.Flatten()(x)

    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Create output layer with a single node and softmax activation
    output = layers.Dense(NUM_CATEGORIES, activation='softmax')(x)

    # Configure and compile the model
    model = Model(img_input, output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    main()