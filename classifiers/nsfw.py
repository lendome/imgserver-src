"""NSFW image classifier using Falconsai/nsfw_image_detection."""

import gc
import logging
from typing import Optional

import torch
from PIL import Image

from .base import BaseClassifier, ClassifierResult

logger = logging.getLogger(__name__)

# NSFW-related keywords for label detection
NSFW_KEYWORDS = {"nsfw", "porn", "sexy", "explicit", "hentai", "adult"}


class NSFWClassifier(BaseClassifier):
    """
    NSFW content detector using Falconsai/nsfw_image_detection.
    
    Runs on CPU to avoid impacting VRAM for image generation.
    Uses ViT (Vision Transformer) architecture.
    """
    
    name = "nsfw"
    model_id = "Falconsai/nsfw_image_detection"
    
    def __init__(self, threshold: float = 0.8, max_image_side: int = 512):
        super().__init__()
        self.threshold = threshold
        self.max_image_side = max_image_side
    
    def load(self) -> None:
        """Load model and processor to CPU."""
        if self._loaded:
            return
        
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        
        logger.info(f"Loading NSFW classifier: {self.model_id}")
        
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self._model.to("cpu")
        self._model.eval()
        
        self._loaded = True
        logger.info("NSFW classifier loaded on CPU")
    
    def unload(self) -> None:
        """Unload model and free memory."""
        if not self._loaded:
            return
        
        logger.info("Unloading NSFW classifier")
        
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
        Check if image contains NSFW content.
        
        Returns:
            ClassifierResult with label ('nsfw' or 'safe'), score (0-1),
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
        
        # Get class labels and find NSFW probability
        id2label = self._model.config.id2label
        
        # Build probability dict
        prob_dict = {}
        nsfw_prob = 0.0
        safe_prob = 0.0
        
        for idx, prob in enumerate(probs.tolist()):
            label = id2label.get(idx, f"class_{idx}").lower()
            prob_dict[label] = prob
            
            # Check if this label is NSFW-related
            if any(kw in label for kw in NSFW_KEYWORDS):
                nsfw_prob = max(nsfw_prob, prob)
            elif "safe" in label or "normal" in label or "sfw" in label:
                safe_prob = max(safe_prob, prob)
        
        # If no explicit safe label found, use 1 - nsfw_prob
        if safe_prob == 0.0:
            safe_prob = 1.0 - nsfw_prob
        
        # Determine result
        is_nsfw = nsfw_prob >= self.threshold
        
        return ClassifierResult(
            label="nsfw" if is_nsfw else "safe",
            score=nsfw_prob,
            raw={
                "probabilities": prob_dict,
                "threshold": self.threshold,
                "is_nsfw": is_nsfw,
            }
        )
    
    def check(self, image: Image.Image) -> bool:
        """Convenience method: returns True if NSFW detected."""
        result = self.predict(image)
        return result.label == "nsfw"
