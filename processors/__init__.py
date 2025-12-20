
"""
Processors package.

Contains image processing modules
"""

__version__ = '1.0.0'

from .image_processor import ImageProcessor
from .image_classifier import ImageClassifier, ClassifierResult
from .coco_object_detector import COCOObjectDetector

__all__ = [
    'ImageProcessor',
    'ImageClassifier',
    'COCOObjectDetector',
    'ClassifierResult',
]

def list_processors() -> list[str]:
    """List available processor modules."""
    return [
        'ImageProcessor',
        'ImageClassifier',
        'COCOObjectDetector',
        'ClassifierResult'
    ]