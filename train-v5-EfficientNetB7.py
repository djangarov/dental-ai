import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from utils import find_problematic_files, save_model, visualize_training


EPOCHS = 50
IMG_WIDTH = 600
IMG_HEIGHT = 600
TEST_SIZE = 0.4
BATCH_SIZE = 32
# BATCH_SIZE = 4  # Much smaller due to high memory requirements


def main():
    parser = argparse.ArgumentParser(description='Train a CNN model on images.')
    parser.add_argument('model_name', help='Model name to save as')
    parser.add_argument('dataset_dir', help='Path to dataset directory for class names')

    args = parser.parse_args()

    # Check if files exist
    if not args.model_name:
        print(f"Error: Model name is required!")
        sys.exit(1)

    if not os.path.exists(args.dataset_dir):
        print(f"Warning: Dataset directory '{args.dataset_dir}' not found! Class names will not be displayed.")

    find_problematic_files(args.dataset_dir)
    # Load image data from directory
    x_train, y_test = load_data(args.dataset_dir)

    # Get actual number of categories from the dataset
    num_categories = len(x_train.class_names)
    print(f"Found {num_categories} categories in dataset")

    # Optimize dataset for performance
    # x_train = x_train.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    # y_test = y_test.prefetch(buffer_size=tf.data.AUTOTUNE)
    x_train = x_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    y_test = y_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Get a compiled neural network
    model = get_model(num_categories)
    model.summary()

    # Fit model on training data
    history = model.fit(x_train, validation_data=y_test, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(y_test, verbose=2)
    visualize_training(args.model_name, history, EPOCHS)

    # Save model to file
    save_model(model, args.model_name)


def load_data(data_dir: str) -> tuple:
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through num_categories - 1. Inside each category directory will be some
    number of image files.
    """
    try:
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

        return x_train, y_test
    except tf.errors.InvalidArgumentError as e:
        print(f"Image format error: {e}")


def get_model(num_categories: int) -> Model:
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    base_model = tf.keras.applications.EfficientNetB7(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False  # Freeze base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_categories, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    main()