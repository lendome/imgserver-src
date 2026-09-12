"""TensorRT compatibility validation for startup checks."""
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from packaging import version

logger = logging.getLogger(__name__)

@dataclass
class CompatibilityResult:
    """Result of compatibility check."""
    compatible: bool
    reason: str
    details: Dict[str, Any]
    warnings: list

def _check_tensorrt_installed() -> Tuple[bool, Optional[str]]:
    """Check if TensorRT is installed and get version."""
    try:
        import tensorrt as trt
        return True, trt.__version__
    except ImportError:
        return False, None

def _check_torch_tensorrt_installed() -> Tuple[bool, Optional[str]]:
    """Check if torch-tensorrt is installed."""
    try:
        import torch_tensorrt
        return True, torch_tensorrt.__version__
    except ImportError:
        return False, None

def _check_onnx_installed() -> Tuple[bool, Optional[str]]:
    """Check if ONNX is installed."""
    try:
        import onnx
        return True, onnx.__version__
    except ImportError:
        return False, None

def _get_cuda_info() -> Dict[str, Any]:
    """Get CUDA information from PyTorch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        
        return {
            "available": True,
            "version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}

def validate_tensorrt_compatibility() -> CompatibilityResult:
    """Comprehensive TensorRT compatibility check.
    
    Returns:
        CompatibilityResult with detailed compatibility information
    """
    warnings = []
    details = {}
    
    # Check CUDA
    cuda_info = _get_cuda_info()
    details["cuda"] = cuda_info
    
    if not cuda_info.get("available"):
        return CompatibilityResult(
            compatible=False,
            reason=f"CUDA not available: {cuda_info.get('reason', 'unknown')}",
            details=details,
            warnings=warnings
        )
    
    # Check CUDA version (need >= 11.8 for TRT 8.6+)
    cuda_version = cuda_info.get("version", "0.0")
    if version.parse(cuda_version) < version.parse("11.8"):
        return CompatibilityResult(
            compatible=False,
            reason=f"CUDA version {cuda_version} < 11.8 required",
            details=details,
            warnings=warnings
        )
    
    # Check compute capability (need SM 7.0+ for good TRT support, SM 8.0+ for Ampere)
    cc = cuda_info.get("compute_capability", (0, 0))
    details["compute_capability"] = cc
    
    if cc < (7, 0):
        return CompatibilityResult(
            compatible=False,
            reason=f"Compute capability {cc[0]}.{cc[1]} < 7.0 required",
            details=details,
            warnings=warnings
        )
    
    if cc < (8, 0):
        warnings.append(f"Compute capability {cc[0]}.{cc[1]} < 8.0 (Ampere). "
                       "TF32 and some optimizations won't be available.")
    
    # Check VRAM (need >= 8GB, recommend >= 12GB)
    vram_gb = cuda_info.get("total_memory_gb", 0)
    details["vram_gb"] = vram_gb
    
    if vram_gb < 8:
        return CompatibilityResult(
            compatible=False,
            reason=f"VRAM {vram_gb:.1f}GB < 8GB minimum required",
            details=details,
            warnings=warnings
        )
    
    if vram_gb < 12:
        warnings.append(f"VRAM {vram_gb:.1f}GB < 12GB recommended. "
                       "May need to use lower precision or smaller batch sizes.")
    
    # Check TensorRT
    trt_installed, trt_version = _check_tensorrt_installed()
    details["tensorrt"] = {"installed": trt_installed, "version": trt_version}
    
    if not trt_installed:
        return CompatibilityResult(
            compatible=False,
            reason="TensorRT not installed. Install with: pip install tensorrt",
            details=details,
            warnings=warnings
        )
    
    if trt_version and version.parse(trt_version) < version.parse("8.6.0"):
        warnings.append(f"TensorRT version {trt_version} < 8.6.0. "
                       "Some features may not be available.")
    
    # Check ONNX (needed for model export)
    onnx_installed, onnx_version = _check_onnx_installed()
    details["onnx"] = {"installed": onnx_installed, "version": onnx_version}
    
    if not onnx_installed:
        warnings.append("ONNX not installed. Required for model export. "
                       "Install with: pip install onnx")
    
    # Check torch-tensorrt (optional, for torch.compile backend)
    torch_trt_installed, torch_trt_version = _check_torch_tensorrt_installed()
    details["torch_tensorrt"] = {"installed": torch_trt_installed, "version": torch_trt_version}
    
    if not torch_trt_installed:
        warnings.append("torch-tensorrt not installed (optional). "
                       "Enables torch.compile with TensorRT backend.")
    
    # Determine compatibility level
    is_rtx3090 = "3090" in cuda_info.get("device_name", "")
    is_ampere = cc[0] == 8
    
    details["optimization_level"] = {
        "is_rtx3090": is_rtx3090,
        "is_ampere": is_ampere,
        "supports_tf32": is_ampere,
        "supports_bf16": is_ampere,
        "optimal_for_tensorrt": is_ampere and vram_gb >= 16,
    }
    
    return CompatibilityResult(
        compatible=True,
        reason="All compatibility checks passed",
        details=details,
        warnings=warnings
    )

def log_compatibility_status() -> bool:
    """Log compatibility status and return whether TensorRT can be used."""
    result = validate_tensorrt_compatibility()
    
    if result.compatible:
        logger.info(f"TensorRT compatibility: OK - {result.reason}")
        
        cuda = result.details.get("cuda", {})
        logger.info(f"  GPU: {cuda.get('device_name', 'unknown')}")
        logger.info(f"  CUDA: {cuda.get('version', 'unknown')}")
        logger.info(f"  Compute Capability: {result.details.get('compute_capability', 'unknown')}")
        logger.info(f"  VRAM: {result.details.get('vram_gb', 0):.1f}GB")
        logger.info(f"  TensorRT: {result.details.get('tensorrt', {}).get('version', 'unknown')}")
        
        opt = result.details.get("optimization_level", {})
        if opt.get("optimal_for_tensorrt"):
            logger.info("  Optimization level: OPTIMAL (Ampere + sufficient VRAM)")
        elif opt.get("is_ampere"):
            logger.info("  Optimization level: GOOD (Ampere architecture)")
        else:
            logger.info("  Optimization level: BASIC (Pre-Ampere)")
    else:
        logger.error(f"TensorRT compatibility: FAILED - {result.reason}")
    
    for warning in result.warnings:
        logger.warning(f"  Warning: {warning}")
    
    return result.compatible

def get_recommended_settings() -> Dict[str, Any]:
    """Get recommended TensorRT settings based on hardware."""
    result = validate_tensorrt_compatibility()
    
    if not result.compatible:
        return {"enabled": False, "reason": result.reason}
    
    vram_gb = result.details.get("vram_gb", 0)
    opt = result.details.get("optimization_level", {})
    
    # Base settings
    settings = {
        "enabled": True,
        "precision": "fp16",
        "fallback_to_pytorch": True,
        "timing_cache_enabled": True,
    }
    
    # Adjust based on VRAM
    if vram_gb >= 20:
        settings["max_workspace_gb"] = 4.0
        settings["builder_optimization_level"] = 4
    elif vram_gb >= 14:
        settings["max_workspace_gb"] = 3.0
        settings["builder_optimization_level"] = 3
    else:
        settings["max_workspace_gb"] = 2.0
        settings["builder_optimization_level"] = 2
    
    # Ampere optimizations
    if opt.get("is_ampere"):
        settings["use_cuda_graph"] = True
    else:
        settings["use_cuda_graph"] = False
    
    return settings
