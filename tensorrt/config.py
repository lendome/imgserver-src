"""TensorRT configuration settings."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


VALID_PRECISIONS = {"fp32", "fp16", "int8"}


@dataclass
class TensorRTSettings:
    """TensorRT acceleration configuration."""
    enabled: bool = False
    precision: str = "fp16"  # fp32, fp16, int8
    max_workspace_gb: float = 4.0
    timing_cache_enabled: bool = True
    builder_optimization_level: int = 3  # 0-5, higher = slower build, faster inference
    use_cuda_graph: bool = True
    dynamic_shapes: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 1
    engine_cache_dir: str = "tensorrt_engines"
    fallback_to_pytorch: bool = True  # Safety fallback
    warmup_iterations: int = 3
    verbose_logging: bool = False
    
    # Resolution profiles for dynamic shapes
    min_height: int = 512
    opt_height: int = 1024
    max_height: int = 1536
    min_width: int = 512
    opt_width: int = 1024
    max_width: int = 1536


def get_engine_cache_path(settings: Optional[TensorRTSettings] = None) -> Path:
    """Returns Path to engine cache directory.
    
    Args:
        settings: TensorRT settings instance. Uses default if not provided.
        
    Returns:
        Path to the engine cache directory.
    """
    if settings is None:
        cache_dir = "tensorrt_engines"
    else:
        cache_dir = settings.engine_cache_dir
    
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_precision(precision: str) -> bool:
    """Validates precision is one of fp32/fp16/int8.
    
    Args:
        precision: The precision string to validate.
        
    Returns:
        True if precision is valid.
        
    Raises:
        ValueError: If precision is not one of the valid options.
    """
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision '{precision}'. Must be one of: {', '.join(sorted(VALID_PRECISIONS))}"
        )
    return True
