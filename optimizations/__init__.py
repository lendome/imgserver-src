"""RTX 3090 Ti optimizations package."""
from .rtx3090 import (
    RTX3090TiConfig,
    apply_rtx3090_optimizations,
    get_rtx3090_memory_config,
    validate_gpu_compatibility,
)

__all__ = [
    "RTX3090TiConfig",
    "apply_rtx3090_optimizations",
    "get_rtx3090_memory_config",
    "validate_gpu_compatibility",
]
