"""Artifact detection for diffusion model outputs."""

import numpy as np
from PIL import Image


def get_image_stats(image: Image.Image) -> dict:
    """Get image statistics for quality analysis."""
    arr = np.array(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA - exclude alpha
        arr = arr[:, :, :3]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def check_image_quality(image: Image.Image) -> tuple[bool, str]:
    """
    Validate image quality and detect common diffusion artifacts.
    
    Returns:
        (is_valid, reason): True with "ok" if clean, False with description if bad.
    """
    arr = np.array(image, dtype=np.float32)
    
    # Check for NaN/Inf pixels
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False, "Image contains NaN or Inf pixel values"
    
    # Handle RGBA - check alpha channel separately
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if np.mean(alpha) < 10:
            return False, "Image is mostly transparent (alpha mean < 10)"
        rgb = arr[:, :, :3]
    elif arr.ndim == 3:
        rgb = arr
    else:
        rgb = arr  # Grayscale
    
    mean_brightness = np.mean(rgb)
    std_dev = np.std(rgb)
    
    # Check all black
    if mean_brightness < 5:
        return False, "Image is all black (mean brightness < 5)"
    
    # Check all white
    if mean_brightness > 250:
        return False, "Image is all white (mean brightness > 250)"
    
    # Check solid color
    if std_dev < 2:
        return False, "Image is solid color (std deviation < 2)"
    
    return True, "ok"
