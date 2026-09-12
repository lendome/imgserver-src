"""CPU-based image classifiers with hot-swapping support."""

from .registry import classifier_registry, get_classifier
from .base import BaseClassifier, ClassifierResult
from .nsfw import NSFWClassifier
from .aesthetic import AestheticClassifier
from .age import AgeClassifier
from .deepface_age import DeepFaceAgeClassifier

# Register classifiers
classifier_registry.register("nsfw", NSFWClassifier)
classifier_registry.register("aesthetic", AestheticClassifier)
classifier_registry.register("age", AgeClassifier)
classifier_registry.register("deepface_age", DeepFaceAgeClassifier)

__all__ = ["classifier_registry", "get_classifier", "BaseClassifier", "ClassifierResult", "NSFWClassifier", "AestheticClassifier", "AgeClassifier", "DeepFaceAgeClassifier"]
