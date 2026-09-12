"""TensorRT acceleration module for image generation."""

from .config import TensorRTSettings, get_engine_cache_path, validate_precision
from .compatibility import (
    CompatibilityResult,
    validate_tensorrt_compatibility,
    log_compatibility_status,
    get_recommended_settings,
)
from .base import TensorRTModuleWrapper, SafeTensorRTExecution
from .cache_manager import (
    EngineMetadata,
    TensorRTEngineCache,
    get_engine_cache,
)
from .fallback import (
    FallbackReason,
    FallbackStats,
    TensorRTFallbackManager,
    get_fallback_manager,
    with_trt_fallback,
    safe_trt_execute,
)
from .validation import (
    ValidationResult,
    TensorRTOutputValidator,
    LatentValidator,
    ImageValidator,
    validate_trt_output,
    validate_latent,
    validate_image,
)
from .sdxl_unet import TRTSDXLUNetWrapper
from .sdxl_vae import (
    SDXL_VAE_SCALING_FACTOR,
    VAEMode,
    TRTSDXLVAEWrapper,
    create_vae_wrapper,
)
from .zimg_transformer import (
    TRTZImageTransformerWrapper,
    BFloat16QualityValidator,
    create_zimage_trt_wrapper,
)
from .benchmark import (
    BenchmarkResult,
    ComparisonResult,
    TensorRTBenchmark,
    create_sample_inputs,
    run_quick_benchmark,
)

__all__ = [
    "TensorRTSettings",
    "get_engine_cache_path",
    "validate_precision",
    "CompatibilityResult",
    "validate_tensorrt_compatibility",
    "log_compatibility_status",
    "get_recommended_settings",
    "TensorRTModuleWrapper",
    "SafeTensorRTExecution",
    "EngineMetadata",
    "TensorRTEngineCache",
    "get_engine_cache",
    "FallbackReason",
    "FallbackStats",
    "TensorRTFallbackManager",
    "get_fallback_manager",
    "with_trt_fallback",
    "safe_trt_execute",
    "ValidationResult",
    "TensorRTOutputValidator",
    "LatentValidator",
    "ImageValidator",
    "validate_trt_output",
    "validate_latent",
    "validate_image",
    "TRTSDXLUNetWrapper",
    "SDXL_VAE_SCALING_FACTOR",
    "VAEMode",
    "TRTSDXLVAEWrapper",
    "create_vae_wrapper",
    "TRTZImageTransformerWrapper",
    "BFloat16QualityValidator",
    "create_zimage_trt_wrapper",
    "BenchmarkResult",
    "ComparisonResult",
    "TensorRTBenchmark",
    "create_sample_inputs",
    "run_quick_benchmark",
]
