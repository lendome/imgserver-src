"""Base classifier interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ClassifierResult:
    """Result from a classifier prediction."""
    
    def __init__(self, label: str, score: float, raw: Optional[dict] = None):
        self.label = label
        self.score = score
        self.raw = raw or {}
    
    def to_dict(self) -> dict:
        return {"label": self.label, "score": self.score, "raw": self.raw}


class BaseClassifier(ABC):
    """Abstract base class for CPU image classifiers."""
    
    name: str = "base"  # Override in subclass
    
    def __init__(self):
        self._model = None
        self._processor = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @abstractmethod
    def load(self) -> None:
        """Load model to CPU. Must set self._loaded = True."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model and free memory. Must set self._loaded = False."""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image) -> ClassifierResult:
        """Run prediction on image. Must be loaded first."""
        pass
    
    def preprocess_image(self, image: Image.Image, max_side: int = 512) -> Image.Image:
        """Resize image to max_side for efficient CPU inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        return image
