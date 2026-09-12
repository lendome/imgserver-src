"""Age classifier using nateraw/vit-age-classifier."""

import gc
import logging
from typing import Optional, Tuple

import torch
from PIL import Image

from .base import BaseClassifier, ClassifierResult

logger = logging.getLogger(__name__)

# Age class definitions from the model
AGE_CLASSES = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]

# Indices that should be considered minor (0-19 years)
MINOR_CLASS_INDICES = {0, 1, 2}  # "0-2", "3-9", "10-19"


class AgeClassifier(BaseClassifier):
    """
    Age classifier using nateraw/vit-age-classifier.
    
    Classifies images into age ranges (0-2, 3-9, 10-19, 20-29, etc.).
    Runs on CPU to avoid impacting VRAM for image generation.
    Uses ViT (Vision Transformer) architecture.
    """
    
    name = "age"
    model_id = "nateraw/vit-age-classifier"
    
    def __init__(self, max_image_side: int = 512):
        super().__init__()
        self.max_image_side = max_image_side
    
    def load(self) -> None:
        """Load model and feature extractor to CPU."""
        if self._loaded:
            return
        
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        
        logger.info(f"Loading age classifier: {self.model_id}")
        
        self._processor = ViTFeatureExtractor.from_pretrained(self.model_id)
        self._model = ViTForImageClassification.from_pretrained(self.model_id)
        self._model.to("cpu")
        self._model.eval()
        
        self._loaded = True
        logger.info("Age classifier loaded on CPU")
    
    def unload(self) -> None:
        """Unload model and free memory."""
        if not self._loaded:
            return
        
        logger.info("Unloading age classifier")
        
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        gc.collect()
        self._loaded = False
    
    def predict(self, image: Image.Image) -> ClassifierResult:
        """
        Predict age class of person in image.
        
        Returns:
            ClassifierResult with label (age class string), score (probability of predicted class),
            and raw dict containing all class probabilities.
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        # Preprocess image
        image = self.preprocess_image(image, self.max_image_side)
        
        # Prepare inputs
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Run inference
        with torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Get predicted class
        predicted_idx = torch.argmax(probs).item()
        predicted_score = probs[predicted_idx].item()
        predicted_label = AGE_CLASSES[predicted_idx]
        
        # Build probability dict for all classes
        prob_dict = {}
        for idx, prob in enumerate(probs.tolist()):
            prob_dict[AGE_CLASSES[idx]] = prob
        
        return ClassifierResult(
            label=predicted_label,
            score=predicted_score,
            raw={
                "probabilities": prob_dict,
                "class_index": predicted_idx,
                "all_classes": AGE_CLASSES,
            }
        )
    
    def check_is_minor(self, image: Image.Image, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Check if person in image appears to be a minor based on age prediction.
        
        Args:
            image: PIL Image to classify
            threshold: Confidence threshold (0-1). Returns True if combined probability
                      of minor classes (0-2, 3-9, 10-19) exceeds this threshold.
        
        Returns:
            Tuple of (is_minor, confidence):
                - is_minor: True if confidence of being minor exceeds threshold
                - confidence: Combined probability of minor age classes
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        result = self.predict(image)
        
        # Calculate combined probability for minor classes
        prob_dict = result.raw["probabilities"]
        minor_confidence = sum(
            prob_dict[AGE_CLASSES[idx]]
            for idx in MINOR_CLASS_INDICES
        )
        
        is_minor = minor_confidence >= threshold
        
        logger.debug(
            f"Age classification: {result.label}, "
            f"minor confidence: {minor_confidence:.3f}, "
            f"is_minor: {is_minor}"
        )
        
        return is_minor, minor_confidence
