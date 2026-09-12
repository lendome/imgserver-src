"""Memory pool configuration for RTX 3090 Ti with TensorRT + PyTorch coexistence."""
import torch
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class VRAMAllocation:
    """VRAM allocation budget for a component."""
    name: str
    min_gb: float
    optimal_gb: float
    max_gb: float
    priority: int  # Lower = higher priority (evict last)

@dataclass  
class RTX3090TiMemoryConfig:
    """Memory configuration optimized for RTX 3090 Ti (24GB VRAM)."""
    
    # Hardware limits
    total_vram_gb: float = 24.0
    reserved_for_system_gb: float = 1.5  # Windows + display overhead
    
    @property
    def usable_vram_gb(self) -> float:
        return self.total_vram_gb - self.reserved_for_system_gb
    
    # TensorRT allocations
    trt_workspace_gb: float = 4.0
    trt_engine_cache_gb: float = 2.0
    trt_timing_cache_mb: float = 64.0
    
    # Model allocations (TensorRT compiled)
    sdxl_unet_trt_gb: float = 2.5       # Reduced from 5GB PyTorch
    sdxl_text_encoders_gb: float = 1.5  # Keep in PyTorch
    sdxl_vae_gb: float = 0.2            # Keep in PyTorch
    
    zimg_transformer_trt_gb: float = 6.0  # Reduced from 12-14GB
    zimg_t5_encoder_gb: float = 3.0       # Keep in PyTorch
    zimg_vae_gb: float = 0.2              # Keep in PyTorch
    
    # Safety margins
    inference_buffer_gb: float = 2.0  # For intermediate activations
    safety_margin_gb: float = 1.0     # Emergency headroom
    
    # L2 Cache optimization (RTX 3090 Ti has 6MB L2)
    l2_cache_mb: float = 6.0
    optimal_tile_size: int = 512  # Optimal for L2 locality

def get_model_allocations() -> Dict[str, VRAMAllocation]:
    """Get VRAM allocation budgets for each model type."""
    return {
        # SDXL components
        "sdxl_unet_pytorch": VRAMAllocation("SDXL UNet (PyTorch)", 4.5, 5.0, 5.5, priority=1),
        "sdxl_unet_trt": VRAMAllocation("SDXL UNet (TRT)", 2.0, 2.5, 3.0, priority=1),
        "sdxl_text_encoder_1": VRAMAllocation("CLIP-L", 0.2, 0.24, 0.3, priority=2),
        "sdxl_text_encoder_2": VRAMAllocation("OpenCLIP-G", 1.2, 1.3, 1.5, priority=2),
        "sdxl_vae": VRAMAllocation("SDXL VAE", 0.15, 0.2, 0.25, priority=3),
        
        # Z-Image components
        "zimg_transformer_pytorch": VRAMAllocation("Z-Image Transformer (PyTorch)", 12.0, 14.0, 16.0, priority=1),
        "zimg_transformer_trt": VRAMAllocation("Z-Image Transformer (TRT)", 5.0, 6.0, 7.0, priority=1),
        "zimg_t5_encoder": VRAMAllocation("T5 Encoder", 2.5, 3.0, 3.5, priority=2),
        "zimg_vae": VRAMAllocation("Z-Image VAE", 0.15, 0.2, 0.25, priority=3),
        
        # TensorRT workspace
        "trt_workspace": VRAMAllocation("TRT Workspace", 2.0, 4.0, 6.0, priority=0),
    }

def calculate_available_vram() -> Dict[str, float]:
    """Calculate current VRAM availability."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    
    return {
        "total_gb": total / (1024**3),
        "allocated_gb": allocated / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "free_gb": (total - reserved) / (1024**3),
        "available_for_allocation_gb": (total - allocated) / (1024**3),
        "fragmentation_ratio": 1.0 - (allocated / reserved) if reserved > 0 else 0.0,
    }

def can_fit_model(model_key: str, with_trt: bool = True) -> Dict[str, any]:
    """Check if a model can fit in current VRAM."""
    allocations = get_model_allocations()
    available = calculate_available_vram()
    
    if "error" in available:
        return {"can_fit": False, "reason": available["error"]}
    
    # Determine which allocation to check
    if with_trt and f"{model_key}_trt" in allocations:
        alloc = allocations[f"{model_key}_trt"]
    elif model_key in allocations:
        alloc = allocations[model_key]
    else:
        return {"can_fit": False, "reason": f"Unknown model: {model_key}"}
    
    free = available["free_gb"]
    
    return {
        "can_fit": free >= alloc.min_gb,
        "comfortable_fit": free >= alloc.optimal_gb,
        "required_gb": alloc.optimal_gb,
        "available_gb": free,
        "headroom_gb": free - alloc.optimal_gb,
    }

def get_optimal_batch_config(available_vram_gb: float) -> Dict[str, int]:
    """Get optimal batch sizes based on available VRAM."""
    # RTX 3090 Ti L2 cache is 6MB, optimize for cache locality
    if available_vram_gb >= 20:
        return {
            "sdxl_batch_size": 1,
            "zimg_batch_size": 1,
            "vae_tile_size": 512,
            "max_latent_size": 192,  # 1536px / 8
        }
    elif available_vram_gb >= 14:
        return {
            "sdxl_batch_size": 1,
            "zimg_batch_size": 1,
            "vae_tile_size": 384,
            "max_latent_size": 160,  # 1280px / 8
        }
    else:
        return {
            "sdxl_batch_size": 1,
            "zimg_batch_size": 1,
            "vae_tile_size": 256,
            "max_latent_size": 128,  # 1024px / 8
        }

def setup_memory_pool(config: Optional[RTX3090TiMemoryConfig] = None) -> None:
    """Configure PyTorch memory pool for optimal TRT coexistence."""
    if config is None:
        config = RTX3090TiMemoryConfig()
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory pool setup")
        return
    
    # Set memory fraction to leave room for TRT
    trt_reserved = config.trt_workspace_gb + config.trt_engine_cache_gb
    pytorch_fraction = (config.usable_vram_gb - trt_reserved) / config.total_vram_gb
    
    # Clamp to reasonable range
    pytorch_fraction = max(0.5, min(0.9, pytorch_fraction))
    
    torch.cuda.set_per_process_memory_fraction(pytorch_fraction, 0)
    logger.info(f"Set PyTorch memory fraction to {pytorch_fraction:.2f} "
                f"(reserving ~{trt_reserved:.1f}GB for TensorRT)")

def log_memory_status():
    """Log current memory status."""
    status = calculate_available_vram()
    if "error" in status:
        logger.warning(f"Cannot get VRAM status: {status['error']}")
        return
    
    logger.info(
        f"VRAM Status: {status['allocated_gb']:.2f}GB allocated, "
        f"{status['free_gb']:.2f}GB free of {status['total_gb']:.2f}GB total "
        f"(fragmentation: {status['fragmentation_ratio']:.1%})"
    )
