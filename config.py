"""Configuration module for the unified image generation server.

Simple config with environment variable support and sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Server configuration with defaults and environment variable support."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Checkpoint paths
    sdxl_checkpoints_dir: str = field(default_factory=lambda: os.getenv(
        "SDXL_CHECKPOINTS_DIR",
        "C:/Users/VIP/Documents/ComfyUI/models/checkpoints"
    ))
    default_checkpoint: str = field(default_factory=lambda: os.getenv(
        "DEFAULT_CHECKPOINT",
        "amanatsuIllustrious_v2.safetensors"
    ))
    
    # Z-Image model (HuggingFace)
    zimage_model_id: str = field(default_factory=lambda: os.getenv(
        "ZIMAGE_MODEL_ID",
        "Tongyi-MAI/Z-Image-Turbo"
    ))
    
    # LoRA cache
    lora_cache_dir: str = field(default_factory=lambda: os.getenv(
        "LORA_CACHE_DIR",
        "./lora_cache"
    ))
    
    # VRAM settings
    vram_safety_margin: float = field(default_factory=lambda: float(os.getenv(
        "VRAM_SAFETY_MARGIN",
        "0.10"
    )))
    
    # Default generation parameters
    default_width: int = field(default_factory=lambda: int(os.getenv(
        "DEFAULT_WIDTH", "1024"
    )))
    default_height: int = field(default_factory=lambda: int(os.getenv(
        "DEFAULT_HEIGHT", "1024"
    )))
    default_steps: int = field(default_factory=lambda: int(os.getenv(
        "DEFAULT_STEPS", "20"
    )))
    default_cfg_scale: float = field(default_factory=lambda: float(os.getenv(
        "DEFAULT_CFG_SCALE", "7.0"
    )))
    default_scheduler: str = field(default_factory=lambda: os.getenv(
        "DEFAULT_SCHEDULER", "euler_a"
    ))
    
    # CUDA settings
    torch_dtype: str = "float16"
    enable_xformers: bool = True

    # torch.compile (Inductor/Triton) acceleration for SDXL UNet.
    # DISABLED BY DEFAULT: compiling costs minutes on the first generation
    # (and again after every server restart), which blocks instant startup.
    # SageAttention + checkpoint hot-swap need no warmup and stay on.
    # Set TORCH_COMPILE_ENABLED=true to opt back into ~+20% speed.
    torch_compile_enabled: bool = field(default_factory=lambda: os.getenv(
        "TORCH_COMPILE_ENABLED", "false"
    ).lower() in ("true", "1", "yes"))

    # SageAttention (INT8 QK quantized attention) for SDXL UNet.
    # Requires the sageattention package + Triton. Falls back to SDPA per-layer
    # for unsupported head dims, or entirely if import fails.
    sage_attention_enabled: bool = field(default_factory=lambda: os.getenv(
        "SAGE_ATTENTION_ENABLED", "true"
    ).lower() in ("true", "1", "yes"))
    
    # TensorRT acceleration settings
    tensorrt_enabled: bool = field(default_factory=lambda: os.getenv(
        "TENSORRT_ENABLED", "true"
    ).lower() in ("true", "1", "yes"))
    tensorrt_engine_cache_dir: str = field(default_factory=lambda: os.getenv(
        "TENSORRT_ENGINE_CACHE_DIR",
        "./tensorrt_engines"
    ))
    
    # Queue settings
    max_queue_size: int = 50  # Maximum pending jobs
    job_retention_seconds: int = 300  # How long to keep completed jobs (5 min)
    request_timeout_seconds: int = 600  # Max time for a single job (10 min)
    concurrent_http_requests: int = 6  # Flask thread pool size hint
    
    def get_checkpoint_path(self, checkpoint_name: Optional[str] = None) -> Path:
        """Get full path to a checkpoint file."""
        name = checkpoint_name or self.default_checkpoint
        return Path(self.sdxl_checkpoints_dir) / name
    
    def get_available_checkpoints(self) -> list[str]:
        """List available checkpoint files."""
        checkpoint_dir = Path(self.sdxl_checkpoints_dir)
        if not checkpoint_dir.exists():
            return []
        return [f.name for f in checkpoint_dir.glob("*.safetensors")]
    
    @property
    def max_vram_usage(self) -> float:
        """Maximum VRAM usage fraction (1.0 - safety margin)."""
        return 1.0 - self.vram_safety_margin


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
