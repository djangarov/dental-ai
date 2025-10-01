import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen

import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')

# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import ops as utils_ops


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if(path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    print(f"Image loaded successfully: {im_width}x{im_height}")

    # Convert to numpy array with correct shape
    image_array = np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

    # Debug: Check if image has actual data
    print(f"Image array shape: {image_array.shape}")
    print(f"Image array min/max: {image_array.min()}/{image_array.max()}")

    return image_array


# COCO class names for better labeling
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

def draw_detections(image, boxes, classes, scores, min_score_thresh=0.3):
    """Draw bounding boxes and labels on the image."""
    print(f"Drawing detections for image with shape: {image.shape}")
    print(f"Image data range: {image.min()} to {image.max()}")

    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return 0

    # Check if image has valid data (not all zeros/white)
    if image.max() == image.min():
        print(f"Warning: Image appears to be uniform (all pixels have value {image.max()})")

    fig, ax = plt.subplots(1, figsize=(12, 8))  # More reasonable size
    ax.imshow(image)

    h, w, _ = image.shape
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    detection_count = 0
    print(f"Processing {len(boxes)} potential detections...")

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if score >= min_score_thresh:
            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to pixel coordinates
            left, top = int(xmin * w), int(ymin * h)
            width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

            color = colors[detection_count % len(colors)]

            # Create rectangle patch
            rect = patches.Rectangle((left, top), width, height,
                                   linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Get class name
            class_name = COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

            # Add label with background
            label = f'{class_name}: {score:.2f}'
            ax.text(left, top - 10, label, fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

            print(f"Detection {detection_count + 1}: {class_name} (Class {int(class_id)}), Score: {score:.3f}")
            print(f"  Box: ({left}, {top}) -> ({left+width}, {top+height})")
            detection_count += 1

    ax.set_title(f'Object Detection Results ({detection_count} objects found)', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return detection_count

def draw_masks(image, boxes, classes, scores, masks, min_score_thresh=0.3):
    """Draw masks if available."""
    print(f"Drawing masks for image with shape: {image.shape}")
    print(f"Image data range: {image.min()} to {image.max()}")
    
    if masks is None:
        print("No masks provided, falling back to bounding boxes")
        return draw_detections(image, boxes, classes, scores, min_score_thresh)

    # Check if image is valid
    if image is None or image.size == 0:
        print("Error: Invalid image data")
        return 0

    # Check if image has valid data (not all zeros/white)
    if image.max() == image.min():
        print(f"Warning: Image appears to be uniform (all pixels have value {image.max()})")
        return 0

    fig, ax = plt.subplots(1, figsize=(12, 8))  # Reduced size

    # Create a copy of the image for mask overlay
    image_with_masks = image.copy().astype(np.float32)

    h, w, _ = image.shape
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

    detection_count = 0
    print(f"Processing {len(masks)} masks...")
    
    for i, (box, score, class_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
        if score >= min_score_thresh:
            print(f"Processing detection {detection_count + 1} with score {score:.3f}")
            
            # Get class name
            class_name = COCO_CLASSES.get(int(class_id), f'Class {int(class_id)}')

            # Apply mask overlay
            color = colors[detection_count % len(colors)]
            
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take first channel if 3D
            
            # Create colored mask
            mask_colored = np.zeros_like(image_with_masks)
            for c in range(3):
                mask_colored[:, :, c] = mask * color[c]

            # Blend mask with image (only where mask > 0.5)
            mask_binary = mask > 0.5
            image_with_masks = np.where(mask_binary[..., None],
                                      0.7 * image_with_masks + 0.3 * mask_colored,
                                      image_with_masks)

            # Draw bounding box
            ymin, xmin, ymax, xmax = box
            left, top = int(xmin * w), int(ymin * h)
            width, height = int((xmax - xmin) * w), int((ymax - ymin) * h)

            rect = patches.Rectangle((left, top), width, height,
                                   linewidth=3, edgecolor=np.array(color)/255, facecolor='none')
            ax.add_patch(rect)

            # Add label
            label = f'{class_name}: {score:.2f}'
            ax.text(left, top - 10, label, fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=np.array(color)/255, alpha=0.8))

            print(f"Detection {detection_count + 1}: {class_name} (Class {int(class_id)}), Score: {score:.3f}")
            print(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            detection_count += 1

    ax.imshow(image_with_masks.astype(np.uint8))
    ax.set_title(f'Object Detection with Masks ({detection_count} objects found)', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return detection_count

# Main execution
if __name__ == "__main__":
    model_handle = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"

    print('loading model...')
    hub_model = hub.load(model_handle)
    print('model loaded!')

    image_path = "./tf-models/research/object_detection/test_images/image1.jpg"
    image_np = load_image_into_numpy_array(image_path)

    # Convert image to grayscale
    # image_np[0] = np.tile(
    #     np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    print("Displaying original image...")
    plt.figure(figsize=(12, 8))  # Smaller size for testing
    plt.imshow(image_np[0])
    plt.title("Original Image", fontsize=16)
    plt.axis('off')
    plt.show()

    # running inference
    print('Running inference...')
    results = hub_model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key: value.numpy() for key, value in results.items()}
    print("Available result keys:", result.keys())

    # Extract detection results
    boxes = result['detection_boxes'][0]
    classes = result['detection_classes'][0]
    scores = result['detection_scores'][0]

    min_score_thresh = 0.3
    print(f"Found {np.sum(scores >= min_score_thresh)} objects with confidence >= {min_score_thresh}")

    valid_detections = np.sum(scores >= min_score_thresh)
    print(f"Found {valid_detections} objects with confidence >= {min_score_thresh}")

    if valid_detections > 0:
    # Draw detections with bounding boxes
        detection_count = draw_detections(image_np[0], boxes, classes, scores, min_score_thresh)

        # Handle models with masks
        if 'detection_masks' in result:
            print("Model supports instance segmentation masks!")
            masks = result['detection_masks'][0]

            # Resize masks to image size
            masks_resized = []
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                if scores[i] >= min_score_thresh:
                    # Get box coordinates
                    ymin, xmin, ymax, xmax = box
                    h, w = image_np.shape[1], image_np.shape[2]

                    # Convert to pixel coordinates
                    y1, x1 = int(ymin * h), int(xmin * w)
                    y2, x2 = int(ymax * h), int(xmax * w)

                    # Resize mask to box size then to full image
                    mask_resized = tf.image.resize(mask[..., None], [y2-y1, x2-x1])
                    mask_full = np.zeros((h, w), dtype=np.float32)
                    mask_full[y1:y2, x1:x2] = mask_resized[:, :, 0].numpy()
                    masks_resized.append(mask_full > 0.5)

            # Draw masks
            if masks_resized:
                valid_masks = [masks_resized[i] for i in range(len(masks_resized)) if scores[i] >= min_score_thresh]
                valid_boxes = boxes[scores >= min_score_thresh]
                valid_classes = classes[scores >= min_score_thresh]
                valid_scores = scores[scores >= min_score_thresh]

                draw_masks(image_np[0], valid_boxes, valid_classes, valid_scores, valid_masks, 0)

    print("Object detection completed!")


