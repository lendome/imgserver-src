"""TensorRT output validation for quality assurance."""
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of output validation."""
    valid: bool
    reason: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

class TensorRTOutputValidator:
    """Validates TensorRT outputs for quality and correctness."""
    
    def __init__(
        self,
        nan_threshold: float = 0.0,  # Any NaN is failure
        inf_threshold: float = 0.0,  # Any Inf is failure
        zero_threshold: float = 0.99,  # >99% zeros is failure
        value_range: Tuple[float, float] = (-1e6, 1e6),  # Reasonable range
    ):
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold
        self.zero_threshold = zero_threshold
        self.value_range = value_range
    
    def validate(self, output: torch.Tensor) -> ValidationResult:
        """Validate a single output tensor."""
        metrics = {}
        
        # Check for NaN
        nan_ratio = torch.isnan(output).float().mean().item()
        metrics["nan_ratio"] = nan_ratio
        if nan_ratio > self.nan_threshold:
            return ValidationResult(
                valid=False,
                reason=f"NaN ratio {nan_ratio:.4f} exceeds threshold {self.nan_threshold}",
                metrics=metrics
            )
        
        # Check for Inf
        inf_ratio = torch.isinf(output).float().mean().item()
        metrics["inf_ratio"] = inf_ratio
        if inf_ratio > self.inf_threshold:
            return ValidationResult(
                valid=False,
                reason=f"Inf ratio {inf_ratio:.4f} exceeds threshold {self.inf_threshold}",
                metrics=metrics
            )
        
        # Check for all zeros (common failure mode)
        zero_ratio = (output == 0).float().mean().item()
        metrics["zero_ratio"] = zero_ratio
        if zero_ratio > self.zero_threshold:
            return ValidationResult(
                valid=False,
                reason=f"Zero ratio {zero_ratio:.4f} exceeds threshold {self.zero_threshold}",
                metrics=metrics
            )
        
        # Check value range
        min_val = output.min().item()
        max_val = output.max().item()
        metrics["min_value"] = min_val
        metrics["max_value"] = max_val
        
        if min_val < self.value_range[0] or max_val > self.value_range[1]:
            return ValidationResult(
                valid=False,
                reason=f"Values [{min_val:.2e}, {max_val:.2e}] outside range {self.value_range}",
                metrics=metrics
            )
        
        # Additional statistics
        metrics["mean"] = output.mean().item()
        metrics["std"] = output.std().item()
        
        return ValidationResult(valid=True, metrics=metrics)
    
    def compare_outputs(
        self,
        trt_output: torch.Tensor,
        pytorch_output: torch.Tensor,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> ValidationResult:
        """Compare TRT output against PyTorch reference."""
        metrics = {}
        
        # Ensure same device for comparison
        if trt_output.device != pytorch_output.device:
            pytorch_output = pytorch_output.to(trt_output.device)
        
        # Ensure same dtype for comparison
        if trt_output.dtype != pytorch_output.dtype:
            trt_output = trt_output.float()
            pytorch_output = pytorch_output.float()
        
        # Compute difference metrics
        diff = (trt_output - pytorch_output).abs()
        
        metrics["max_abs_diff"] = diff.max().item()
        metrics["mean_abs_diff"] = diff.mean().item()
        
        # Relative difference (avoid division by zero)
        rel_diff = diff / (pytorch_output.abs() + atol)
        metrics["max_rel_diff"] = rel_diff.max().item()
        metrics["mean_rel_diff"] = rel_diff.mean().item()
        
        # Check if outputs are close
        is_close = torch.allclose(trt_output, pytorch_output, rtol=rtol, atol=atol)
        
        # PSNR for image-like outputs
        if trt_output.ndim >= 3:
            mse = F.mse_loss(trt_output.float(), pytorch_output.float())
            if mse > 0:
                psnr = 10 * torch.log10(1.0 / mse)
                metrics["psnr"] = psnr.item()
        
        if not is_close:
            return ValidationResult(
                valid=False,
                reason=f"Outputs differ: max_abs_diff={metrics['max_abs_diff']:.2e}, "
                       f"max_rel_diff={metrics['max_rel_diff']:.2e}",
                metrics=metrics
            )
        
        return ValidationResult(valid=True, metrics=metrics)

class LatentValidator(TensorRTOutputValidator):
    """Specialized validator for diffusion model latents."""
    
    def __init__(self):
        super().__init__(
            nan_threshold=0.0,
            inf_threshold=0.0,
            zero_threshold=0.95,  # Latents can have many zeros
            value_range=(-100.0, 100.0),  # Latents are usually in smaller range
        )
    
    def validate_latent(self, latent: torch.Tensor) -> ValidationResult:
        """Validate latent tensor from diffusion model."""
        result = self.validate(latent)
        
        if not result.valid:
            return result
        
        # Additional latent-specific checks
        metrics = result.metrics or {}
        
        # Check expected shape [B, 4, H, W] for SDXL
        if latent.ndim == 4 and latent.shape[1] != 4:
            return ValidationResult(
                valid=False,
                reason=f"Unexpected latent channels: {latent.shape[1]} (expected 4)",
                metrics=metrics
            )
        
        # Check reasonable latent statistics
        std = latent.std().item()
        if std < 0.01:
            logger.warning(f"Latent has very low variance (std={std:.4f})")
        
        return ValidationResult(valid=True, metrics=metrics)

class ImageValidator(TensorRTOutputValidator):
    """Specialized validator for decoded images."""
    
    def __init__(self):
        super().__init__(
            nan_threshold=0.0,
            inf_threshold=0.0,
            zero_threshold=0.5,  # Images shouldn't be mostly black
            value_range=(-1.0, 1.0),  # Normalized image range
        )
    
    def validate_image(self, image: torch.Tensor) -> ValidationResult:
        """Validate decoded image tensor."""
        result = self.validate(image)
        
        if not result.valid:
            return result
        
        metrics = result.metrics or {}
        
        # Check expected shape [B, 3, H, W]
        if image.ndim == 4 and image.shape[1] != 3:
            return ValidationResult(
                valid=False,
                reason=f"Unexpected image channels: {image.shape[1]} (expected 3)",
                metrics=metrics
            )
        
        # Check for mostly black or white images
        mean = image.mean().item()
        if abs(mean) > 0.9:
            logger.warning(f"Image appears mostly {'white' if mean > 0 else 'black'} (mean={mean:.3f})")
        
        return ValidationResult(valid=True, metrics=metrics)

# Convenience functions
def validate_trt_output(output: torch.Tensor) -> bool:
    """Quick validation check for TRT output."""
    validator = TensorRTOutputValidator()
    result = validator.validate(output)
    if not result.valid:
        logger.warning(f"TRT output validation failed: {result.reason}")
    return result.valid

def validate_latent(latent: torch.Tensor) -> bool:
    """Quick validation for latent tensor."""
    validator = LatentValidator()
    result = validator.validate_latent(latent)
    if not result.valid:
        logger.warning(f"Latent validation failed: {result.reason}")
    return result.valid

def validate_image(image: torch.Tensor) -> bool:
    """Quick validation for image tensor."""
    validator = ImageValidator()
    result = validator.validate_image(image)
    if not result.valid:
        logger.warning(f"Image validation failed: {result.reason}")
    return result.valid
