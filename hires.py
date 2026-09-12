"""Hires fix module for SDXL image upscaling and refinement."""

from dataclasses import dataclass
from PIL import Image
import torch


@dataclass
class HiresConfig:
    """Configuration for hires fix processing."""
    enabled: bool = False
    upscale_by: float = 1.5
    denoising_strength: float | None = None  # Auto-calc if None


def calculate_hires_denoising(upscale_by: float) -> float:
    """Calculate optimal denoising strength based on upscale factor.
    
    Args:
        upscale_by: The upscale multiplier (e.g., 1.5 for 1.5x)
        
    Returns:
        Recommended denoising strength value.
    """
    if upscale_by <= 1.2:
        return 0.45
    elif upscale_by <= 1.5:
        return 0.55
    elif upscale_by <= 1.8:
        return 0.62
    elif upscale_by <= 2.0:
        return 0.68
    else:
        return 0.75


def apply_hires_fix(
    base_image: Image.Image,
    pipeline,
    prompt: str,
    negative_prompt: str,
    config: HiresConfig,
    seed: int | None,
    steps: int,
    cfg_scale: float,
) -> Image.Image:
    """Apply hires fix to upscale and refine an image.
    
    Args:
        base_image: The base image to upscale and refine.
        pipeline: SDXL pipeline with img2img capability.
        prompt: Generation prompt.
        negative_prompt: Negative prompt.
        config: HiresConfig with upscale settings.
        seed: Random seed for reproducibility.
        steps: Number of inference steps.
        cfg_scale: Classifier-free guidance scale.
        
    Returns:
        Refined high-resolution image.
    """
    if not config.enabled:
        return base_image
    
    # Calculate target size
    orig_width, orig_height = base_image.size
    new_width = int(orig_width * config.upscale_by)
    new_height = int(orig_height * config.upscale_by)
    
    # Upscale with Lanczos interpolation
    upscaled_image = base_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Determine denoising strength
    denoising = config.denoising_strength
    if denoising is None:
        denoising = calculate_hires_denoising(config.upscale_by)
    
    # Setup generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Run img2img refinement
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=upscaled_image,
        strength=denoising,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    
    return result.images[0]
