"""Aesthetic quality classifier using LAION aesthetic predictor."""

import gc
import logging
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .base import BaseClassifier, ClassifierResult

logger = logging.getLogger(__name__)


class AestheticMLP(nn.Module):
    """Simple MLP for aesthetic scoring (from LAION)."""
    
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        return self.layers(x)


class AestheticClassifier(BaseClassifier):
    """
    Aesthetic quality scorer using CLIP + LAION aesthetic predictor.
    
    Scores images on a 1-10 scale for visual appeal/quality.
    Higher scores indicate more aesthetically pleasing images.
    
    Uses:
    - CLIP ViT-L/14 for image embeddings
    - LAION aesthetic predictor MLP for scoring
    """
    
    name = "aesthetic"
    clip_model_id = "openai/clip-vit-large-patch14"
    aesthetic_model_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
    
    def __init__(self, max_image_side: int = 512):
        super().__init__()
        self.max_image_side = max_image_side
        self._clip_model = None
        self._clip_processor = None
        self._aesthetic_model = None
    
    def load(self) -> None:
        """Load CLIP and aesthetic models to CPU."""
        if self._loaded:
            return
        
        from transformers import CLIPModel, CLIPProcessor
        import urllib.request
        import tempfile
        import os
        
        logger.info("Loading aesthetic classifier (CLIP + MLP)")
        
        # Load CLIP model
        logger.debug("Loading CLIP model...")
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_id)
        self._clip_model.to("cpu")
        self._clip_model.eval()
        
        # Load aesthetic MLP
        logger.debug("Loading aesthetic MLP...")
        self._aesthetic_model = AestheticMLP(input_size=768)
        
        # Download weights if needed
        cache_dir = os.path.join(tempfile.gettempdir(), "aesthetic_cache")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "aesthetic_predictor.pth")
        
        if not os.path.exists(weights_path):
            logger.debug(f"Downloading aesthetic weights to {weights_path}")
            urllib.request.urlretrieve(self.aesthetic_model_url, weights_path)
        
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._aesthetic_model.load_state_dict(state_dict)
        self._aesthetic_model.to("cpu")
        self._aesthetic_model.eval()
        
        self._loaded = True
        logger.info("Aesthetic classifier loaded on CPU")
    
    def unload(self) -> None:
        """Unload models and free memory."""
        if not self._loaded:
            return
        
        logger.info("Unloading aesthetic classifier")
        
        if self._clip_model is not None:
            del self._clip_model
            self._clip_model = None
        if self._clip_processor is not None:
            del self._clip_processor
            self._clip_processor = None
        if self._aesthetic_model is not None:
            del self._aesthetic_model
            self._aesthetic_model = None
        
        gc.collect()
        self._loaded = False
    
    def predict(self, image: Image.Image) -> ClassifierResult:
        """
        Score image aesthetics on 1-10 scale.
        
        Returns:
            ClassifierResult with label (quality tier), score (1-10),
            and raw dict containing normalized score.
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        # Preprocess image
        image = self.preprocess_image(image, self.max_image_side)
        
        # Get CLIP embeddings
        inputs = self._clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.inference_mode():
            # Get image features from CLIP
            image_features = self._clip_model.get_image_features(**inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get aesthetic score
            score = self._aesthetic_model(image_features).item()
        
        # Clamp to 1-10 range (model can sometimes output outside)
        score = max(1.0, min(10.0, score))
        
        # Determine quality label
        if score >= 7.0:
            label = "excellent"
        elif score >= 5.5:
            label = "good"
        elif score >= 4.0:
            label = "average"
        elif score >= 2.5:
            label = "poor"
        else:
            label = "bad"
        
        return ClassifierResult(
            label=label,
            score=score,
            raw={
                "aesthetic_score": score,
                "quality_tier": label,
                "scale": "1-10",
            }
        )
    
    def score(self, image: Image.Image) -> float:
        """Convenience method: returns aesthetic score (1-10)."""
        result = self.predict(image)
        return result.score
