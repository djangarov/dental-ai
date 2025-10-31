import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import sys
import argparse

from predictor import Predictor

def preprocess_image(image_path: str, img_size: tuple) -> tf.Tensor:
    """
    Preprocess a single image for prediction
    """
    # Read the image file
    img = keras.utils.load_img(
        image_path, target_size=img_size
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

def main():
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('model_path', help='Path to the trained model file (.keras)')
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('dataset', help='Path to dataset directory for class names')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f"Warning: Dataset directory '{args.dataset}' not found! Class names will not be displayed.")

    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    print("Model loaded successfully!")

    predictor = Predictor(model, args.dataset)

    # Preprocess the image
    print(f"Processing image: {args.image_path}")
    processed_image = preprocess_image(args.image_path, model.input_shape[1:3])

    # Make prediction
    predictor.predict_image(processed_image)

if __name__ == "__main__":
    main()