import os
import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.keras import Model
import matplotlib.pyplot as plt


def validate_image_format(image_path: str) -> bool:
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


def find_problematic_files(data_dir: str) -> list[str]:
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


def visualize_training(history: History, epochs: int) -> None:
    """
    Visualize training history
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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

    save_path = 'training_plot.png'
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Training plot saved to {save_path}")

    plt.show()

def save_model(model: Model, model_name: str) -> None:
    """
    Save the trained model to a file
    """
    filename = f"{model_name}.keras"
    model.save(filename)
    print(f"Model saved to {filename}.")