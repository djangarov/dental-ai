import argparse
import os
import sys
from custom_trainer import CustomTrainer
from inception_resnet_v2_trainer import InceptionResNetV2Trainer
from vgg19_trainer import VGG19Trainer
from resnet50_trainer import ResNet50Trainer
from inception_v3_trainer import InceptionV3Trainer


def main():
    parser = argparse.ArgumentParser(description='Train CNN models on images.')
    parser.add_argument('model_type', choices=['vgg19', 'resnet50', 'inceptionv3', 'custom', 'inception_resnet_v2'], help='Type of model to train')
    parser.add_argument('model_name', help='Model name to save as')
    parser.add_argument('dataset_dir', help='Path to dataset directory')

    args = parser.parse_args()

    # Check if files exist
    if not args.model_name:
        print(f"Error: Model name is required!")
        sys.exit(1)

    if not os.path.exists(args.dataset_dir):
        print(f"Warning: Dataset directory '{args.dataset_dir}' not found! Class names will not be displayed.")

    # Create trainer based on model type
    trainers = {
        'vgg19': VGG19Trainer(),
        'resnet50': ResNet50Trainer(),
        'inceptionv3': InceptionV3Trainer(),
        'custom': CustomTrainer(),
        'inception_resnet_v2': InceptionResNetV2Trainer()
    }

    trainer = trainers[args.model_type]

    # Train the model
    model, history = trainer.train(args.dataset_dir, args.model_name)

    print(f"Training completed for {args.model_type}")


if __name__ == "__main__":
    main()