"""VRAM pre-flight checks for generation requests."""
import logging
from .vram import vram_manager

logger = logging.getLogger(__name__)

# VRAM limits
MAX_VRAM_USAGE = 0.95  # 95% of total
SAFETY_MARGIN = 0.90   # 90% of free

def estimate_generation_vram(width: int, height: int, steps: int = 20, hires: bool = False) -> int:
    """Estimate VRAM needed for generation in bytes."""
    # Base model VRAM (~7GB for SDXL)
    base = 7 * 1024**3
    
    # Latent size (width/8 * height/8 * 4 channels * 2 bytes fp16 * 3 copies)
    latent_size = (width // 8) * (height // 8) * 4 * 2 * 3
    
    # Attention memory (~150MB per megapixel)
    megapixels = (width * height) / 1_000_000
    attention = int(megapixels * 150 * 1024**2)
    
    # Step overhead
    step_overhead = steps * 10 * 1024**2  # ~10MB per step
    
    total = base + latent_size + attention + step_overhead
    
    if hires:
        total = int(total * 1.5)  # Hires needs more
    
    return total

def check_vram_bounds(width: int, height: int, steps: int = 20, hires: bool = False) -> tuple[bool, str]:
    """
    Check if generation parameters fit in VRAM.
    
    Returns:
        (can_generate, message): True if OK, False with reason if not
    """
    total_vram = vram_manager.get_total()
    free_vram = vram_manager.get_free()
    estimated = estimate_generation_vram(width, height, steps, hires)
    
    # Check against total VRAM limit
    max_allowed = int(total_vram * MAX_VRAM_USAGE)
    current_used = vram_manager.get_used()
    
    if current_used + estimated > max_allowed:
        return False, f"Would exceed {MAX_VRAM_USAGE*100:.0f}% VRAM limit"
    
    # Check against free VRAM with safety margin
    safe_free = int(free_vram * SAFETY_MARGIN)
    if estimated > safe_free:
        return False, f"Need {estimated/1e9:.2f}GB but only {safe_free/1e9:.2f}GB safely available"
    
    return True, "OK"

def suggest_safe_params(width: int, height: int) -> dict:
    """Suggest safer parameters if current ones are too large."""
    free_vram = vram_manager.get_free()
    
    # If less than 10GB free, suggest smaller resolution
    if free_vram < 10 * 1024**3:
        scale = 0.75
        return {
            "width": int(width * scale // 8) * 8,
            "height": int(height * scale // 8) * 8,
            "reason": "Low VRAM, reduced resolution"
        }
    return {"width": width, "height": height, "reason": None}
