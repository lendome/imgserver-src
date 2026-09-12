"""DeepFace age classifier using RetinaFace detector backend."""

import gc
import io
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from .base import BaseClassifier, ClassifierResult

# Optimize for multi-core CPUs (e.g., Ryzen 9 5950x)
# Pin threads to physical cores to prevent hyper-threading overhead
os.environ["OMP_NUM_THREADS"] = "16"

logger = logging.getLogger(__name__)


class DeepFaceAgeClassifier(BaseClassifier):
    """
    Age classifier using DeepFace library with RetinaFace detector backend.
    
    Uses RetinaFace for SOTA face detection and DeepFace's age model for estimation.
    Runs on CPU only to avoid impacting VRAM for image generation.
    Supports multiple face detection in a single image.
    """
    
    name = "deepface_age"
    
    def __init__(self, adult_threshold: int = 18, max_image_side: int = 1024):
        """
        Initialize the DeepFace age classifier.
        
        Args:
            adult_threshold: Age threshold for adult classification (default 18)
            max_image_side: Maximum image dimension for preprocessing
        """
        super().__init__()
        self.adult_threshold = adult_threshold
        self.max_image_side = max_image_side
        self._deepface = None
    
    def load(self) -> None:
        """Load DeepFace models to CPU and keep them in RAM."""
        if self._loaded:
            return
        
        logger.info("Loading DeepFace age classifier with RetinaFace backend")
        
        # Import DeepFace
        from deepface import DeepFace
        self._deepface = DeepFace
        
        # Pre-build and cache the age model using DeepFace's public API
        logger.info("Pre-loading age estimation model...")
        self._deepface.build_model(model_name="Age", task="facial_attribute")
        
        # Pre-build and cache the RetinaFace detector
        logger.info("Pre-loading RetinaFace detector...")
        self._deepface.build_model(model_name="retinaface", task="face_detector")
        
        self._loaded = True
        logger.info("DeepFace age classifier loaded and cached in RAM")
    
    def unload(self) -> None:
        """Unload DeepFace and free memory."""
        if not self._loaded:
            return
        
        logger.info("Unloading DeepFace age classifier")
        
        if self._deepface is not None:
            self._deepface = None
        
        gc.collect()
        self._loaded = False
    
    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR numpy array)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def _analyze_image(self, img: np.ndarray, enforce_detection: bool = True) -> List[dict]:
        """
        Analyze image using DeepFace with pre-loaded models.
        
        Args:
            img: OpenCV image (BGR numpy array)
            enforce_detection: If True, raises error when no face detected
            
        Returns:
            List of face analysis results
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        # Models are cached internally by DeepFace after build_model calls
        results = self._deepface.analyze(
            img_path=img,
            actions=['age'],
            detector_backend='retinaface',
            enforce_detection=enforce_detection,
            align=True,
            silent=True
        )
        
        # DeepFace returns a list if multiple faces, single dict otherwise
        if not isinstance(results, list):
            results = [results]
        
        return results
    
    def predict(self, image: Image.Image) -> ClassifierResult:
        """
        Predict age of person(s) in image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            ClassifierResult with:
                - label: "adult" or "minor" based on primary face
                - score: face detection confidence
                - raw: dict with estimated_age, is_adult, face_confidence, 
                       bounding_box, all_faces
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        # Preprocess image
        image = self.preprocess_image(image, self.max_image_side)
        
        # Convert to OpenCV format
        cv_img = self._pil_to_cv2(image)
        
        try:
            results = self._analyze_image(cv_img, enforce_detection=True)
        except ValueError:
            # No face detected
            return ClassifierResult(
                label="unknown",
                score=0.0,
                raw={
                    "error": "No face detected in image",
                    "estimated_age": None,
                    "is_adult": None,
                    "face_confidence": 0.0,
                    "bounding_box": None,
                    "all_faces": []
                }
            )
        except Exception as e:
            logger.error(f"DeepFace analysis error: {e}")
            return ClassifierResult(
                label="error",
                score=0.0,
                raw={
                    "error": str(e),
                    "estimated_age": None,
                    "is_adult": None,
                    "face_confidence": 0.0,
                    "bounding_box": None,
                    "all_faces": []
                }
            )
        
        # Process all detected faces
        all_faces = []
        for face in results:
            est_age = face['age']
            face_confidence = face.get('face_confidence', 1.0)
            bounding_box = face.get('region', {})
            
            all_faces.append({
                "estimated_age": est_age,
                "is_adult": est_age >= self.adult_threshold,
                "face_confidence": face_confidence,
                "bounding_box": bounding_box
            })
        
        # Use first/primary face for main result
        primary = all_faces[0]
        label = "adult" if primary["is_adult"] else "minor"
        
        return ClassifierResult(
            label=label,
            score=primary["face_confidence"],
            raw={
                "estimated_age": primary["estimated_age"],
                "is_adult": primary["is_adult"],
                "face_confidence": primary["face_confidence"],
                "bounding_box": primary["bounding_box"],
                "all_faces": all_faces
            }
        )
    
    def predict_from_url(self, url: str) -> ClassifierResult:
        """
        Predict age from image URL.
        
        Args:
            url: URL of image to analyze
            
        Returns:
            ClassifierResult with age estimation results
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        try:
            # Download image
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Decode image
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            cv_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if cv_img is None:
                return ClassifierResult(
                    label="error",
                    score=0.0,
                    raw={
                        "error": "Failed to decode image from URL",
                        "estimated_age": None,
                        "is_adult": None,
                        "face_confidence": 0.0,
                        "bounding_box": None,
                        "all_faces": []
                    }
                )
            
            # Convert to PIL for consistent processing
            cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_rgb)
            
            return self.predict(pil_image)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download image from URL: {e}")
            return ClassifierResult(
                label="error",
                score=0.0,
                raw={
                    "error": f"Failed to download image: {str(e)}",
                    "estimated_age": None,
                    "is_adult": None,
                    "face_confidence": 0.0,
                    "bounding_box": None,
                    "all_faces": []
                }
            )
    
    def check_adult(
        self, 
        image: Image.Image, 
        adult_threshold: Optional[int] = None
    ) -> Tuple[bool, int, float]:
        """
        Check if the primary face in image is an adult.
        
        Args:
            image: PIL Image to analyze
            adult_threshold: Age threshold for adult classification
                           (defaults to instance threshold if not provided)
            
        Returns:
            Tuple of (is_adult, estimated_age, confidence):
                - is_adult: True if estimated age >= threshold
                - estimated_age: Predicted age in years
                - confidence: Face detection confidence score
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        threshold = adult_threshold if adult_threshold is not None else self.adult_threshold
        
        result = self.predict(image)
        
        if result.label in ("unknown", "error"):
            # No face detected or error - return safe defaults
            return False, 0, 0.0
        
        est_age = result.raw["estimated_age"]
        confidence = result.raw["face_confidence"]
        is_adult = est_age >= threshold
        
        logger.debug(
            f"DeepFace age check: estimated_age={est_age}, "
            f"is_adult={is_adult}, confidence={confidence:.3f}"
        )
        
        return is_adult, est_age, confidence
    
    def check_adult_from_url(
        self, 
        url: str, 
        adult_threshold: Optional[int] = None
    ) -> Tuple[bool, int, float]:
        """
        Check if the primary face in image URL is an adult.
        
        Args:
            url: URL of image to analyze
            adult_threshold: Age threshold for adult classification
            
        Returns:
            Tuple of (is_adult, estimated_age, confidence)
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call load() first.")
        
        result = self.predict_from_url(url)
        
        if result.label in ("unknown", "error"):
            return False, 0, 0.0
        
        threshold = adult_threshold if adult_threshold is not None else self.adult_threshold
        est_age = result.raw["estimated_age"]
        confidence = result.raw["face_confidence"]
        is_adult = est_age >= threshold
        
        return is_adult, est_age, confidence
    
    def get_all_faces(self, image: Image.Image) -> List[dict]:
        """
        Get age estimation for all faces detected in image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of dicts, each containing:
                - estimated_age: Predicted age
                - is_adult: Whether age >= adult_threshold
                - face_confidence: Detection confidence
                - bounding_box: Face region coordinates
        """
        result = self.predict(image)
        return result.raw.get("all_faces", [])
