"""TensorRT wrapper for SDXL VAE encode/decode operations.

This module provides TensorRT acceleration for SDXL VAE with:
- Support for both encode and decode modes
- Automatic FP32 upcast fallback for problematic models
- VAE slicing compatibility (fallback to PyTorch)
- Dynamic shape support for various resolutions

Note: This is lower priority than UNet TensorRT as VAE is typically
faster and less impactful to overall generation time.
"""

import logging
import os
from enum import Enum
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

from .base import TensorRTModuleWrapper
from .config import TensorRTSettings, get_engine_cache_path

logger = logging.getLogger(__name__)


# SDXL VAE scaling factor
SDXL_VAE_SCALING_FACTOR = 0.13025


class VAEMode(Enum):
    """VAE operation mode."""
    ENCODE = "encode"
    DECODE = "decode"


class TRTSDXLVAEWrapper(TensorRTModuleWrapper):
    """TensorRT wrapper for SDXL VAE encoder/decoder.
    
    Supports both encoding (image → latents) and decoding (latents → image)
    with automatic fallback to PyTorch when:
    - TensorRT compilation fails
    - FP32 upcast is required
    - VAE slicing is enabled
    - Input dimensions exceed TRT profile limits
    
    Args:
        vae_module: The PyTorch VAE module (AutoencoderKL or similar)
        mode: VAEMode.ENCODE or VAEMode.DECODE
        settings: TensorRT configuration settings
        force_upcast: Force FP32 precision for VAE (some models need this)
        device: Target device ("cuda" or specific GPU)
    """
    
    def __init__(
        self,
        vae_module: nn.Module,
        mode: VAEMode = VAEMode.DECODE,
        settings: Optional[TensorRTSettings] = None,
        force_upcast: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            pytorch_module=vae_module,
            fallback_enabled=True,
            device=device,
        )
        
        self.mode = mode
        self.settings = settings or TensorRTSettings()
        self.force_upcast = force_upcast
        self.scaling_factor = SDXL_VAE_SCALING_FACTOR
        
        # Track slicing state - when enabled, always use PyTorch
        self._slicing_enabled = False
        self._tiling_enabled = False
        
        # FP32 upcast detection
        self._requires_fp32 = force_upcast or self._detect_fp32_requirement()
        
        # Skip TRT compilation if FP32 is required (TRT FP16 would produce bad results)
        if self._requires_fp32 and self.settings.precision == "fp16":
            logger.info("VAE requires FP32 - TensorRT disabled for this module")
            self._compilation_failed = True
    
    def _detect_fp32_requirement(self) -> bool:
        """Detect if VAE requires FP32 precision for correct output.
        
        Some VAE models (especially older ones) produce NaN/artifacts
        when run in FP16. This checks model config for hints.
        """
        if hasattr(self.pytorch_module, 'config'):
            config = self.pytorch_module.config
            # Check for force_upcast flag in config
            if hasattr(config, 'force_upcast') and config.force_upcast:
                return True
            # Some models specify sample_size that hints at precision needs
            if hasattr(config, 'scaling_factor'):
                # Models with non-standard scaling may need FP32
                pass
        
        # Check if module is already in FP32
        try:
            first_param = next(self.pytorch_module.parameters())
            if first_param.dtype == torch.float32:
                logger.debug("VAE is in FP32, may require upcast")
        except StopIteration:
            pass
        
        return False
    
    @property
    def slicing_enabled(self) -> bool:
        """Check if VAE slicing is enabled."""
        return self._slicing_enabled
    
    @slicing_enabled.setter
    def slicing_enabled(self, value: bool):
        """Set VAE slicing state - disables TRT when enabled."""
        self._slicing_enabled = value
        if value:
            logger.debug("VAE slicing enabled - TensorRT will be bypassed")
    
    @property
    def tiling_enabled(self) -> bool:
        """Check if VAE tiling is enabled."""
        return self._tiling_enabled
    
    @tiling_enabled.setter
    def tiling_enabled(self, value: bool):
        """Set VAE tiling state - disables TRT when enabled."""
        self._tiling_enabled = value
        if value:
            logger.debug("VAE tiling enabled - TensorRT will be bypassed")
    
    def enable_slicing(self):
        """Enable VAE slicing (for memory efficiency)."""
        self.slicing_enabled = True
        if hasattr(self.pytorch_module, 'enable_slicing'):
            self.pytorch_module.enable_slicing()
    
    def disable_slicing(self):
        """Disable VAE slicing."""
        self.slicing_enabled = False
        if hasattr(self.pytorch_module, 'disable_slicing'):
            self.pytorch_module.disable_slicing()
    
    def enable_tiling(self):
        """Enable VAE tiling (for very high resolutions)."""
        self.tiling_enabled = True
        if hasattr(self.pytorch_module, 'enable_tiling'):
            self.pytorch_module.enable_tiling()
    
    def disable_tiling(self):
        """Disable VAE tiling."""
        self.tiling_enabled = False
        if hasattr(self.pytorch_module, 'disable_tiling'):
            self.pytorch_module.disable_tiling()
    
    def get_input_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return input tensor specifications for TensorRT."""
        settings = self.settings
        
        if self.mode == VAEMode.DECODE:
            # Decoder input: latent space [B, 4, H/8, W/8]
            return {
                "latent_sample": {
                    "dtype": torch.float16 if settings.precision == "fp16" else torch.float32,
                    "min_shape": (1, 4, settings.min_height // 8, settings.min_width // 8),
                    "opt_shape": (1, 4, settings.opt_height // 8, settings.opt_width // 8),
                    "max_shape": (1, 4, settings.max_height // 8, settings.max_width // 8),
                }
            }
        else:
            # Encoder input: image space [B, 3, H, W]
            return {
                "sample": {
                    "dtype": torch.float16 if settings.precision == "fp16" else torch.float32,
                    "min_shape": (1, 3, settings.min_height, settings.min_width),
                    "opt_shape": (1, 3, settings.opt_height, settings.opt_width),
                    "max_shape": (1, 3, settings.max_height, settings.max_width),
                }
            }
    
    def get_output_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return output tensor specifications for TensorRT."""
        settings = self.settings
        dtype = torch.float16 if settings.precision == "fp16" else torch.float32
        
        if self.mode == VAEMode.DECODE:
            # Decoder output: image space [B, 3, H, W]
            return {
                "sample": {
                    "dtype": dtype,
                    "min_shape": (1, 3, settings.min_height, settings.min_width),
                    "opt_shape": (1, 3, settings.opt_height, settings.opt_width),
                    "max_shape": (1, 3, settings.max_height, settings.max_width),
                }
            }
        else:
            # Encoder output: latent space [B, 4, H/8, W/8]
            return {
                "latent_sample": {
                    "dtype": dtype,
                    "min_shape": (1, 4, settings.min_height // 8, settings.min_width // 8),
                    "opt_shape": (1, 4, settings.opt_height // 8, settings.opt_width // 8),
                    "max_shape": (1, 4, settings.max_height // 8, settings.max_width // 8),
                }
            }
    
    def _export_to_onnx(self, output_path: str, sample_inputs: Dict[str, torch.Tensor]) -> bool:
        """Export VAE to ONNX format for TensorRT compilation."""
        try:
            import torch.onnx
            
            logger.info(f"Exporting VAE {self.mode.value} to ONNX: {output_path}")
            
            # Prepare module for export
            self.pytorch_module.eval()
            
            if self.mode == VAEMode.DECODE:
                # Export decoder
                decoder = self.pytorch_module.decoder
                sample = sample_inputs["latent_sample"]
                input_names = ["latent_sample"]
                output_names = ["sample"]
                dynamic_axes = {
                    "latent_sample": {0: "batch", 2: "height", 3: "width"},
                    "sample": {0: "batch", 2: "height_out", 3: "width_out"},
                }
                
                # Create a wrapper to handle post_quant_conv + decoder
                class DecoderWrapper(nn.Module):
                    def __init__(self, vae):
                        super().__init__()
                        self.post_quant_conv = vae.post_quant_conv
                        self.decoder = vae.decoder
                    
                    def forward(self, latent_sample):
                        z = self.post_quant_conv(latent_sample)
                        return self.decoder(z)
                
                export_module = DecoderWrapper(self.pytorch_module)
                
            else:
                # Export encoder
                sample = sample_inputs["sample"]
                input_names = ["sample"]
                output_names = ["latent_sample"]
                dynamic_axes = {
                    "sample": {0: "batch", 2: "height", 3: "width"},
                    "latent_sample": {0: "batch", 2: "height_out", 3: "width_out"},
                }
                
                # Create wrapper for encoder + quant_conv
                class EncoderWrapper(nn.Module):
                    def __init__(self, vae):
                        super().__init__()
                        self.encoder = vae.encoder
                        self.quant_conv = vae.quant_conv
                    
                    def forward(self, sample):
                        h = self.encoder(sample)
                        moments = self.quant_conv(h)
                        # Return mean (first half of channels)
                        mean, _ = torch.chunk(moments, 2, dim=1)
                        return mean
                
                export_module = EncoderWrapper(self.pytorch_module)
            
            export_module.eval()
            
            with torch.no_grad():
                torch.onnx.export(
                    export_module,
                    (sample,),
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True,
                )
            
            logger.info("ONNX export successful")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def _build_trt_engine(self, onnx_path: str) -> bool:
        """Build TensorRT engine from ONNX model."""
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(
                trt.Logger.VERBOSE if self.settings.verbose_logging else trt.Logger.WARNING
            )
            
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(i)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                int(self.settings.max_workspace_gb * (1 << 30))
            )
            
            # Set precision
            if self.settings.precision == "fp16" and not self._requires_fp32:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Optimization level
            config.builder_optimization_level = self.settings.builder_optimization_level
            
            # Dynamic shapes profile
            profile = builder.create_optimization_profile()
            input_spec = self.get_input_spec()
            
            for name, spec in input_spec.items():
                profile.set_shape(
                    name,
                    spec["min_shape"],
                    spec["opt_shape"],
                    spec["max_shape"],
                )
            
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("TensorRT engine build failed")
                return False
            
            # Deserialize and store
            runtime = trt.Runtime(TRT_LOGGER)
            self._trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            self._trt_context = self._trt_engine.create_execution_context()
            
            # Cache engine to disk
            self._cache_engine(serialized_engine)
            
            logger.info("TensorRT engine built successfully")
            return True
            
        except ImportError:
            logger.error("TensorRT not available - install tensorrt package")
            return False
        except Exception as e:
            logger.error(f"TensorRT engine build failed: {e}")
            return False
    
    def _cache_engine(self, serialized_engine: bytes) -> None:
        """Save serialized engine to cache directory."""
        try:
            cache_dir = get_engine_cache_path(self.settings)
            
            mode_str = self.mode.value
            precision = "fp32" if self._requires_fp32 else self.settings.precision
            
            cache_file = cache_dir / f"sdxl_vae_{mode_str}_{precision}.engine"
            
            with open(cache_file, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"Cached TensorRT engine: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache TensorRT engine: {e}")
    
    def load_cached_engine(self) -> bool:
        """Load TensorRT engine from cache if available."""
        try:
            import tensorrt as trt
            
            cache_dir = get_engine_cache_path(self.settings)
            mode_str = self.mode.value
            precision = "fp32" if self._requires_fp32 else self.settings.precision
            
            cache_file = cache_dir / f"sdxl_vae_{mode_str}_{precision}.engine"
            
            if not cache_file.exists():
                logger.debug(f"No cached engine found: {cache_file}")
                return False
            
            TRT_LOGGER = trt.Logger(
                trt.Logger.VERBOSE if self.settings.verbose_logging else trt.Logger.WARNING
            )
            
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(cache_file, 'rb') as f:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())
            
            if self._trt_engine is None:
                logger.warning("Failed to deserialize cached engine")
                return False
            
            self._trt_context = self._trt_engine.create_execution_context()
            self._allocate_buffers()
            self._trt_stream = torch.cuda.Stream()
            self._compiled = True
            
            logger.info(f"Loaded cached TensorRT engine: {cache_file}")
            return True
            
        except ImportError:
            logger.debug("TensorRT not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to load cached engine: {e}")
            return False
    
    def _trt_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Execute TensorRT inference."""
        if self.mode == VAEMode.DECODE:
            latent_sample = inputs.get("latent_sample")
            if latent_sample is None:
                raise ValueError("latent_sample input required for decode mode")
            
            # Set input shape
            self._trt_context.set_input_shape("latent_sample", tuple(latent_sample.shape))
            
            # Prepare output buffer
            batch, _, h, w = latent_sample.shape
            output_shape = (batch, 3, h * 8, w * 8)
            output = torch.empty(output_shape, dtype=latent_sample.dtype, device=self.device)
            
            # Bind tensors
            self._trt_context.set_tensor_address("latent_sample", latent_sample.data_ptr())
            self._trt_context.set_tensor_address("sample", output.data_ptr())
            
            # Execute
            self._trt_context.execute_async_v3(self._trt_stream.cuda_stream)
            
            return {"sample": output}
            
        else:
            sample = inputs.get("sample")
            if sample is None:
                raise ValueError("sample input required for encode mode")
            
            # Set input shape
            self._trt_context.set_input_shape("sample", tuple(sample.shape))
            
            # Prepare output buffer
            batch, _, h, w = sample.shape
            output_shape = (batch, 4, h // 8, w // 8)
            output = torch.empty(output_shape, dtype=sample.dtype, device=self.device)
            
            # Bind tensors
            self._trt_context.set_tensor_address("sample", sample.data_ptr())
            self._trt_context.set_tensor_address("latent_sample", output.data_ptr())
            
            # Execute
            self._trt_context.execute_async_v3(self._trt_stream.cuda_stream)
            
            return {"latent_sample": output}
    
    def _pytorch_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Execute PyTorch inference (fallback)."""
        if self.mode == VAEMode.DECODE:
            latent_sample = inputs.get("latent_sample")
            if latent_sample is None:
                raise ValueError("latent_sample input required for decode mode")
            
            # Handle FP32 upcast if needed
            original_dtype = latent_sample.dtype
            if self._requires_fp32:
                latent_sample = latent_sample.to(torch.float32)
                self.pytorch_module.to(torch.float32)
            
            # Decode
            with torch.no_grad():
                sample = self.pytorch_module.decode(latent_sample).sample
            
            # Restore dtype if needed
            if self._requires_fp32 and original_dtype != torch.float32:
                sample = sample.to(original_dtype)
            
            return {"sample": sample}
            
        else:
            sample = inputs.get("sample")
            if sample is None:
                raise ValueError("sample input required for encode mode")
            
            # Handle FP32 upcast if needed
            original_dtype = sample.dtype
            if self._requires_fp32:
                sample = sample.to(torch.float32)
                self.pytorch_module.to(torch.float32)
            
            # Encode
            with torch.no_grad():
                latent_dist = self.pytorch_module.encode(sample).latent_dist
                latent_sample = latent_dist.sample()
            
            # Restore dtype if needed
            if self._requires_fp32 and original_dtype != torch.float32:
                latent_sample = latent_sample.to(original_dtype)
            
            return {"latent_sample": latent_sample}
    
    def forward(
        self,
        sample: Optional[torch.Tensor] = None,
        latent_sample: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with automatic TRT/PyTorch selection.
        
        For decode mode: provide latent_sample
        For encode mode: provide sample
        
        Args:
            sample: Input image tensor [B, 3, H, W] (for encode)
            latent_sample: Latent tensor [B, 4, H/8, W/8] (for decode)
            return_dict: If True, return dict; otherwise return tensor
            
        Returns:
            Decoded image or encoded latents
        """
        # Check for slicing/tiling - force PyTorch fallback
        if self._slicing_enabled or self._tiling_enabled:
            logger.debug("Slicing/tiling enabled - using PyTorch")
            if self.mode == VAEMode.DECODE:
                result = self._pytorch_forward(latent_sample=latent_sample)
            else:
                result = self._pytorch_forward(sample=sample)
        else:
            # Use parent's forward logic (TRT with fallback)
            if self.mode == VAEMode.DECODE:
                result = super().forward(latent_sample=latent_sample)
            else:
                result = super().forward(sample=sample)
        
        if return_dict:
            return result
        
        # Return just the tensor
        if self.mode == VAEMode.DECODE:
            return result.get("sample")
        else:
            return result.get("latent_sample")
    
    def decode(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode latents to image.
        
        Convenience method that handles scaling factor.
        
        Args:
            latents: Latent tensor [B, 4, H/8, W/8]
            return_dict: If True, return dict
            
        Returns:
            Decoded image tensor [B, 3, H, W]
        """
        if self.mode != VAEMode.DECODE:
            logger.warning("decode() called but wrapper is in encode mode")
        
        # Apply inverse scaling
        scaled_latents = latents / self.scaling_factor
        
        return self.forward(latent_sample=scaled_latents, return_dict=return_dict)
    
    def encode(
        self,
        image: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode image to latents.
        
        Convenience method that handles scaling factor.
        
        Args:
            image: Image tensor [B, 3, H, W]
            return_dict: If True, return dict
            
        Returns:
            Encoded latent tensor [B, 4, H/8, W/8]
        """
        if self.mode != VAEMode.ENCODE:
            logger.warning("encode() called but wrapper is in decode mode")
        
        result = self.forward(sample=image, return_dict=True)
        
        # Apply scaling
        latents = result.get("latent_sample")
        if latents is not None:
            latents = latents * self.scaling_factor
            result["latent_sample"] = latents
        
        if return_dict:
            return result
        return latents
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get extended performance statistics."""
        stats = super().get_performance_stats()
        stats.update({
            "mode": self.mode.value,
            "force_upcast": self.force_upcast,
            "requires_fp32": self._requires_fp32,
            "slicing_enabled": self._slicing_enabled,
            "tiling_enabled": self._tiling_enabled,
            "scaling_factor": self.scaling_factor,
        })
        return stats


def create_vae_wrapper(
    vae_module: nn.Module,
    mode: str = "decode",
    settings: Optional[TensorRTSettings] = None,
    force_upcast: bool = False,
    auto_compile: bool = False,
    device: str = "cuda",
) -> TRTSDXLVAEWrapper:
    """Factory function to create VAE TensorRT wrapper.
    
    Args:
        vae_module: PyTorch VAE module
        mode: "encode" or "decode"
        settings: TensorRT settings
        force_upcast: Force FP32 precision
        auto_compile: Automatically compile TRT engine
        device: Target device
        
    Returns:
        Configured TRTSDXLVAEWrapper instance
    """
    vae_mode = VAEMode.ENCODE if mode == "encode" else VAEMode.DECODE
    
    wrapper = TRTSDXLVAEWrapper(
        vae_module=vae_module,
        mode=vae_mode,
        settings=settings,
        force_upcast=force_upcast,
        device=device,
    )
    
    if auto_compile:
        # Try loading cached engine first
        if not wrapper.load_cached_engine():
            wrapper.compile()
    
    return wrapper
