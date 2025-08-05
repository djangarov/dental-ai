from custom_trainer import CustomTrainer
from efficientnet_b7_trainer import EfficientNetB7Trainer
from inception_resnet_v2_trainer import InceptionResNetV2Trainer
from vgg19_trainer import VGG19Trainer
from resnet50_trainer import ResNet50Trainer
from inception_v3_trainer import InceptionV3Trainer


class ModelFactory:
    """
    Factory class for creating model trainers
    """

    @staticmethod
    def create_trainer(model_type: str):
        """
        Create trainer instance based on model type
        """
        trainers = {
            'vgg19': VGG19Trainer,
            'resnet50': ResNet50Trainer,
            'inceptionv3': InceptionV3Trainer,
            'custom': CustomTrainer,
            'inception_resnet_v2': InceptionResNetV2Trainer,
            'efficientnet_b7': EfficientNetB7Trainer
        }

        if model_type not in trainers:
            raise ValueError(f"Unknown model type: {model_type}")

        return trainers[model_type]()

    @staticmethod
    def get_available_models():
        """
        Get list of available model types
        """
        return ['vgg19', 'resnet50', 'inceptionv3', 'custom', 'inception_resnet_v2', 'efficientnet_b7']