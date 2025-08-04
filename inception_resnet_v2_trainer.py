import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from base_trainer import BaseTrainer


class InceptionResNetV2Trainer(BaseTrainer):
    """
    InceptionResNetV2 model trainer
    """

    def __init__(self):
        super().__init__(
            model_name="InceptionResNetV2",
            epochs=1,
            test_size=0.4,
            batch_size=32,
            # batch_size=16, # Reduced due to larger image size
            image_width=299,
            image_height=299)  # InceptionV3 specific settings

    def get_model(self, num_categories: int) -> Model:
        """
        Returns InceptionResNetV2 transfer learning model
        """
        # InceptionResNetV2 base model
        base_model = tf.keras.applications.InceptionResNetV2(
            input_shape=(self.img_width, self.img_height, 3),
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