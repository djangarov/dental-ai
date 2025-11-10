import keras

from trainers import BaseTrainer


class ResNet50Trainer(BaseTrainer):
    """
    ResNet50 model trainer
    """

    def __init__(self):
        super().__init__(
            model_name='ResNet50',
            epochs=50,
            test_size=0.4,
            # batch_size=32,
            batch_size=16, # Reduced due to larger image size
            image_width=224,
            image_height=224)  # VGG19 specific settings

    def get_model(self, num_categories: int) -> keras.Model:
        """
        Returns RestNet50 transfer learning model
        """
        # RestNet50 base model
        base_model = keras.applications.ResNet50(
            input_shape=(self.img_width, self.img_height, 3),
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False  # Freeze base model

        inputs = keras.Input(shape=(self.img_width, self.img_height, 3))

        # Data augmentation
        x = keras.layers.RandomFlip('horizontal')(inputs)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
        x = keras.layers.RandomBrightness(0.1)(x)

        # ResNet50 preprocessing
        x = keras.applications.resnet50.preprocess_input(x)
        x = base_model(x, training=False)

        # Add a named convolutional layer for Grad-CAM access
        x = keras.layers.Conv2D(
            512, (1, 1),
            activation='relu',
            name='grad_cam_conv'
        )(x)

        # Classification head
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)

        outputs = keras.layers.Dense(num_categories, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model