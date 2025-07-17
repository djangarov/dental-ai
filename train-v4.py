import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

EPOCHS =50
IMG_WIDTH = 150
IMG_HEIGHT = 150
TEST_SIZE = 0.4
BATCH_SIZE = 32


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

    # For small datasets, use more aggressive strategies
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,  # More patience for small datasets
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{args.model_name}_best.keras",
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    # Fit model on training data
    history = model.fit(
        x_train,
        validation_data=y_test,
        epochs=EPOCHS,  # Increased epochs
        callbacks=callbacks
    )
    # Evaluate neural network performance
    model.evaluate(y_test, verbose=2)
    visualize_training(history)

    # # Save model to file
    filename = args.model_name + ".keras"
    model.save(filename)
    print(f"Model saved to {filename}.")

def validate_image_format(image_path):
    """
    Validate if an image can be decoded by TensorFlow
    """
    try:
        # Read the image file
        image_raw = tf.io.read_file(image_path)

        # Try to decode the image
        image = tf.image.decode_image(image_raw, channels=3)

        # Check if image has valid dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False

        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False

def find_problematic_files(data_dir):
    """
    Find files that might cause issues including broken images
    """
    problematic_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Check for files without extensions or with unusual extensions
            if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                problematic_files.append(file_path)
                print(f"Unsupported format: {file_path}")
                os.remove(file_path)
                continue

            # Validate image format using TensorFlow
            if not validate_image_format(file_path):
                problematic_files.append(file_path)
                print(f"Broken image found and removed: {file_path}")
                os.remove(file_path)

    return problematic_files

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
    img_input = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Data augmentation using the following Keras preprocessing layers
    x = layers.RandomFlip('horizontal')(img_input)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
    x = layers.Rescaling(1./255)(x)

    # First convolution extracts 32 filters that are 3x3
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)  # Early dropout

    # Second convolution extracts 64 filters that are 3x3
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Third convolution extracts 128 filters that are 3x3
    x = layers.Convolution2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Fourth convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Flatten feature map to a 1-dim tensor
    x = layers.GlobalAveragePooling2D()(x)

    # Smaller dense layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Create output layer with a single node and softmax activation
    output = layers.Dense(num_categories, activation='softmax')(x)

    # Configure and compile the model
    model = Model(img_input, output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

def visualize_training(history):
    """
    Visualize training history
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()