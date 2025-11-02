
"""
Trainers package.

Contains predictor module
"""

__version__ = "1.0.0"

from .image_predictor import ImagePredictor

__all__ = [
    "ImagePredictor",
]

def list_predictors():
    """List available predictor modules."""
    return [
        "ImagePredictor",
    ]