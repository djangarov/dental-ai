import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Predictor():
    def __init__(self, model: keras.Model, dataset_dir: str | None = None):
        self.model = model
        self.dataset_dir = dataset_dir

    def __load_class_names(self) -> list | None:
        if self.dataset_dir is not None and not os.path.exists(self.dataset_dir):
            return None

        class_names = []
        for class_dir in sorted(os.listdir(self.dataset_dir)):
            if os.path.isdir(os.path.join(self.dataset_dir, class_dir)):
                class_names.append(class_dir)

        return class_names

    def predict_image(self, image: tf.Tensor):
        """
        Predict the class of a single image
        """
        try:
            # Get class names if dataset directory is provided
            class_names = self.__load_class_names()

            # Make prediction
            print("Making prediction...")
            predictions = self.model.predict(image, verbose=0)

            # Get the predicted class
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]

            # Get top 5 predictions
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            top_5_confidences = predictions[0][top_5_indices]

            # Print results
            print("\n" + "="*50)
            print(f"PREDICTION RESULTS")
            print("="*50)
            print(f"Predicted class ID: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

            if class_names and predicted_class < len(class_names):
                print(f"Predicted class name: {class_names[predicted_class]}")
                print("\nTop 5 predictions:")
                for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
                    if idx < len(class_names):
                        print(f"{i+1}. {class_names[idx]}: {conf:.4f} ({conf*100:.2f}%)")
            else:
                print("\nTop 5 predictions:")
                for i, (idx, conf) in enumerate(zip(top_5_indices, top_5_confidences)):
                    print(f"{i+1}. Class {idx}: {conf:.4f} ({conf*100:.2f}%)")

            return predicted_class, confidence

        except Exception as e:
            print(f"Error during prediction: {str(e)}")

            return None, None