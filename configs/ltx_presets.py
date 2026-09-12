"""LTX Video generation presets for quality/speed tradeoffs."""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LTXPreset:
    """Configuration preset for LTX video generation."""
    name: str
    description: str
    
    # Inference settings
    num_inference_steps: int
    guidance_scale: float
    
    # Enhanced guidance (from LTX-2 architecture)
    stg_scale: float = 0.0  # Spatio-Temporal Guidance
    stg_blocks: Optional[List[int]] = None
    rescale_scale: float = 0.0
    
    # Performance options
    use_gradient_estimation: bool = False
    ge_gamma: float = 2.0
    
    # Decode settings
    decode_timestep: float = 0.05
    decode_noise_scale: Optional[float] = None
    
    # Memory optimization
    fp8_enabled: bool = False
    smart_decode: bool = True


# Define presets
QUALITY_PRESET = LTXPreset(
    name="quality",
    description="Highest quality, slowest inference (~40 steps)",
    num_inference_steps=40,
    guidance_scale=3.5,
    stg_scale=1.0,
    stg_blocks=[29],
    rescale_scale=0.7,
    decode_timestep=0.05,
)

BALANCED_PRESET = LTXPreset(
    name="balanced",
    description="Good quality/speed balance (~30 steps)",
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=0.5,
    stg_blocks=[29],
    rescale_scale=0.5,
    decode_timestep=0.05,
)

FAST_PRESET = LTXPreset(
    name="fast",
    description="Fastest inference with gradient estimation (~20 steps)",
    num_inference_steps=20,
    guidance_scale=2.5,
    stg_scale=0.0,
    use_gradient_estimation=True,
    ge_gamma=2.0,
    decode_timestep=0.03,
)

MEMORY_EFFICIENT_PRESET = LTXPreset(
    name="memory_efficient",
    description="Optimized for low VRAM with FP8 (~30 steps, ~8GB VRAM)",
    num_inference_steps=30,
    guidance_scale=3.0,
    fp8_enabled=True,
    smart_decode=True,
)

# Preset lookup
PRESETS = {
    "quality": QUALITY_PRESET,
    "balanced": BALANCED_PRESET,
    "fast": FAST_PRESET,
    "memory_efficient": MEMORY_EFFICIENT_PRESET,
}


def get_preset(name: str) -> LTXPreset:
    """Get a preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
