import keras

from trainers import BaseTrainer


class VGG19Trainer(BaseTrainer):
    """
    VGG19 model trainer
    """

    def __init__(self) -> None:
        super().__init__(
            model_name='VGG19',
            epochs=50,
            batch_size=32,  # Adjusted for VGG19 memory requirements
            image_width=224,
            image_height=224)  # VGG19 specific settings

    def get_model(self, num_categories: int) -> keras.Model:
        """
        Returns VGG19 transfer learning model with data augmentation.

        Args:
            num_categories: Number of output classes for classification

        Returns:
            Compiled Keras model with frozen VGG19 base and custom head
        """
        # VGG19 base model pre-trained on ImageNet
        base_model = keras.applications.VGG19(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False  # Freeze base model for transfer learning

        # Add preprocessing and data augmentation
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))

        # Data augmentation layers
        x = keras.layers.RandomFlip('horizontal')(inputs)
        x = keras.layers.RandomRotation(0.15)(x)
        x = keras.layers.RandomZoom(0.15)(x)
        x = keras.layers.RandomBrightness(0.1)(x)

        # VGG19 preprocessing and feature extraction
        x = keras.applications.vgg19.preprocess_input(x)
        x = base_model(x, training=False)

        # Add a named convolutional layer for Grad-CAM visualization
        x = keras.layers.Conv2D(
            512, (1, 1),
            activation='relu',
            name='grad_cam_conv'
        )(x)

        # Classification head with regularization
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)

        # Output layer for multi-class classification
        outputs = keras.layers.Dense(num_categories, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Compile model with lower learning rate for transfer learning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model