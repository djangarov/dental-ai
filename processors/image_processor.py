from abc import ABC

import numpy as np
import tensorflow as tf
import keras


class ImageProcessor(ABC):
    def preprocess_image(self, image_path: str, image_size: tuple[int, int] | None = None) -> tf.Tensor:
        """
        Preprocess a single image for prediction.

        Args:
            image_path: Path to the image file
            image_size: Target size as (height, width) tuple, or None for original size

        Returns:
            Preprocessed image tensor ready for model inference
        """
        image = keras.utils.load_img(
            image_path, target_size=image_size
        )
        image_array = keras.utils.img_to_array(image, dtype=np.uint8)
        image_array = tf.expand_dims(image_array, axis=0)

        return image_array
