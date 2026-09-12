"""RTX 3090 Ti specific optimizations for Ampere architecture."""
import os
import torch
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class RTX3090TiConfig:
    """RTX 3090 Ti (Ampere GA102) specific settings."""
    # CUDA Memory Allocator
    cuda_alloc_config: str = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256"
    
    # Compute optimizations
    tf32_enabled: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    float32_matmul_precision: str = "high"  # high/medium/highest
    
    # TensorRT specific for Ampere
    trt_dla_enabled: bool = False  # Not available on consumer GPUs
    trt_sparse_weights: bool = False  # Requires model retraining
    trt_timing_cache: bool = True
    
    # Tensor Core settings
    use_tensor_cores: bool = True
    prefer_cudnn_v8: bool = True
    
    # Hardware specs (for reference)
    vram_gb: float = 24.0
    memory_bus_width: int = 384
    memory_bandwidth_gbps: float = 936.0
    cuda_cores: int = 10752
    tensor_cores: int = 336
    compute_capability: tuple = (8, 6)  # SM 8.6
    l2_cache_mb: float = 6.0

def apply_rtx3090_optimizations(config: Optional[RTX3090TiConfig] = None) -> None:
    """Apply RTX 3090 Ti specific CUDA optimizations."""
    if config is None:
        config = RTX3090TiConfig()
    
    # Set CUDA allocator config (must be before CUDA init)
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.cuda_alloc_config
        logger.info(f"Set CUDA allocator config: {config.cuda_alloc_config}")
    
    # TF32 for Ampere tensor cores (8x faster matmul)
    if config.tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for Ampere tensor cores")
    
    # cuDNN settings
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    
    # Float32 matmul precision
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(config.float32_matmul_precision)
        logger.info(f"Set float32 matmul precision: {config.float32_matmul_precision}")

def get_rtx3090_memory_config() -> dict:
    """Get optimal memory configuration for RTX 3090 Ti."""
    return {
        "total_vram_gb": 24.0,
        "reserved_for_system_gb": 1.5,  # Windows/display overhead
        "usable_vram_gb": 22.5,
        
        # Safe thresholds
        "warning_threshold": 0.85,
        "critical_threshold": 0.92,
        "oom_prevention_threshold": 0.95,
        
        # Optimal settings
        "max_split_size_mb": 256,
        "gc_threshold": 0.8,
    }

def validate_gpu_compatibility() -> dict:
    """Check if current GPU is RTX 3090 Ti compatible."""
    if not torch.cuda.is_available():
        return {"compatible": False, "reason": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    
    return {
        "compatible": cc >= (8, 0),  # Ampere or newer
        "gpu_name": props.name,
        "compute_capability": cc,
        "is_ampere": cc[0] == 8,
        "is_rtx3090": "3090" in props.name,
        "vram_gb": props.total_memory / (1024**3),
        "supports_tf32": cc >= (8, 0),
        "supports_bf16": cc >= (8, 0),
    }
