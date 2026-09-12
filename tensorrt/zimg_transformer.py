"""TensorRT wrapper for Z-Image Transformer with bfloat16 conversion handling."""

import logging
import tempfile
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from .base import TensorRTModuleWrapper
from .validation import TensorRTOutputValidator, ValidationResult

logger = logging.getLogger(__name__)

# Shape constants for Z-Image (resolution / 8 for latent space)
MIN_LATENT_SIZE = 64    # 512 / 8
OPT_LATENT_H = 96       # 768 / 8
OPT_LATENT_W = 128      # 1024 / 8
MAX_LATENT_H = 128      # 1024 / 8
MAX_LATENT_W = 192      # 1536 / 8

# T5 embedding specs
T5_SEQ_LENGTH = 512
T5_HIDDEN_DIM = 768


class BFloat16QualityValidator(TensorRTOutputValidator):
    """Validates FP16 TRT output against bfloat16 reference for quality degradation."""
    
    def __init__(
        self,
        max_relative_error: float = 0.05,  # 5% max relative error
        min_psnr: float = 35.0,            # Minimum PSNR in dB
        cosine_threshold: float = 0.995,    # Cosine similarity threshold
    ):
        super().__init__()
        self.max_relative_error = max_relative_error
        self.min_psnr = min_psnr
        self.cosine_threshold = cosine_threshold
    
    def validate_bf16_to_fp16(
        self,
        fp16_output: torch.Tensor,
        bf16_reference: torch.Tensor,
    ) -> ValidationResult:
        """Compare FP16 TRT output against bfloat16 PyTorch reference.
        
        Args:
            fp16_output: FP16 output from TensorRT
            bf16_reference: bfloat16 reference from PyTorch
            
        Returns:
            ValidationResult with quality metrics
        """
        metrics = {}
        
        # Convert both to float32 for comparison
        fp16_f32 = fp16_output.float()
        bf16_f32 = bf16_reference.float()
        
        # Ensure same device
        if fp16_f32.device != bf16_f32.device:
            bf16_f32 = bf16_f32.to(fp16_f32.device)
        
        # Mean relative error
        rel_error = (fp16_f32 - bf16_f32).abs() / (bf16_f32.abs() + 1e-8)
        mean_rel_error = rel_error.mean().item()
        max_rel_error = rel_error.max().item()
        metrics["mean_relative_error"] = mean_rel_error
        metrics["max_relative_error"] = max_rel_error
        
        # PSNR
        mse = ((fp16_f32 - bf16_f32) ** 2).mean()
        if mse > 0:
            max_val = max(bf16_f32.abs().max().item(), 1.0)
            psnr = 10 * torch.log10(max_val ** 2 / mse).item()
            metrics["psnr"] = psnr
        else:
            psnr = float('inf')
            metrics["psnr"] = psnr
        
        # Cosine similarity (flattened)
        fp16_flat = fp16_f32.flatten()
        bf16_flat = bf16_f32.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            fp16_flat.unsqueeze(0), bf16_flat.unsqueeze(0)
        ).item()
        metrics["cosine_similarity"] = cosine_sim
        
        # Quality checks
        if mean_rel_error > self.max_relative_error:
            return ValidationResult(
                valid=False,
                reason=f"Mean relative error {mean_rel_error:.4f} exceeds threshold {self.max_relative_error}",
                metrics=metrics
            )
        
        if psnr < self.min_psnr:
            return ValidationResult(
                valid=False,
                reason=f"PSNR {psnr:.1f}dB below threshold {self.min_psnr}dB",
                metrics=metrics
            )
        
        if cosine_sim < self.cosine_threshold:
            return ValidationResult(
                valid=False,
                reason=f"Cosine similarity {cosine_sim:.4f} below threshold {self.cosine_threshold}",
                metrics=metrics
            )
        
        return ValidationResult(valid=True, metrics=metrics)


class TRTZImageTransformerWrapper(TensorRTModuleWrapper):
    """TensorRT wrapper for Z-Image Transformer with bfloat16 handling.
    
    Z-Image uses bfloat16 natively, but TensorRT doesn't support bfloat16.
    This wrapper handles conversion at TRT boundaries with quality validation.
    
    Key features:
    - Automatic bfloat16 to FP16 conversion at input
    - FP16 to bfloat16 conversion at output  
    - Quality validation to detect precision degradation
    - Automatic fallback to PyTorch if quality drops
    - Flow matching compatible (continuous timesteps)
    """
    
    def __init__(
        self,
        pytorch_module: nn.Module,
        fallback_enabled: bool = True,
        device: str = "cuda",
        quality_threshold: float = 0.05,  # Max 5% quality degradation
        validation_frequency: int = 100,  # Validate every N calls
    ):
        super().__init__(pytorch_module, fallback_enabled, device)
        
        self.quality_threshold = quality_threshold
        self.validation_frequency = validation_frequency
        self._quality_validator = BFloat16QualityValidator(
            max_relative_error=quality_threshold
        )
        
        # Quality tracking
        self._validation_call_count = 0
        self._quality_failures = 0
        self._last_quality_metrics: Optional[Dict[str, float]] = None
        self._quality_degraded = False
    
    def get_input_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return input tensor specifications for Z-Image transformer.
        
        Inputs:
        - hidden_states: latent [batch, 4, H/8, W/8]
        - encoder_hidden_states: T5 embeddings [batch, 512, 768]
        - timestep: continuous float for flow matching
        """
        return {
            "hidden_states": {
                "dtype": torch.float16,  # Converted from bfloat16
                "min_shape": (1, 4, MIN_LATENT_SIZE, MIN_LATENT_SIZE),
                "opt_shape": (1, 4, OPT_LATENT_H, OPT_LATENT_W),
                "max_shape": (2, 4, MAX_LATENT_H, MAX_LATENT_W),
            },
            "encoder_hidden_states": {
                "dtype": torch.float16,
                "min_shape": (1, T5_SEQ_LENGTH, T5_HIDDEN_DIM),
                "opt_shape": (1, T5_SEQ_LENGTH, T5_HIDDEN_DIM),
                "max_shape": (2, T5_SEQ_LENGTH, T5_HIDDEN_DIM),
            },
            "timestep": {
                "dtype": torch.float32,  # Keep timestep as float32 for precision
                "min_shape": (1,),
                "opt_shape": (1,),
                "max_shape": (2,),
            },
        }
    
    def get_output_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return output tensor specifications."""
        return {
            "output": {
                "dtype": torch.float16,  # Will convert back to bfloat16
                "min_shape": (1, 4, MIN_LATENT_SIZE, MIN_LATENT_SIZE),
                "opt_shape": (1, 4, OPT_LATENT_H, OPT_LATENT_W),
                "max_shape": (2, 4, MAX_LATENT_H, MAX_LATENT_W),
            }
        }
    
    def _convert_bf16_to_fp16(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert bfloat16 tensor to float16 for TensorRT.
        
        bfloat16 has same exponent range as float32 but less precision.
        float16 has smaller exponent range but similar precision to bf16.
        Some values may overflow/underflow - we clamp to be safe.
        """
        if tensor.dtype == torch.bfloat16:
            # Convert via float32 to avoid precision loss
            fp32 = tensor.float()
            
            # Clamp to FP16 range to avoid overflow
            fp16_max = 65504.0  # Max representable in FP16
            fp32 = fp32.clamp(-fp16_max, fp16_max)
            
            return fp32.half()
        elif tensor.dtype == torch.float32:
            return tensor.half()
        return tensor
    
    def _convert_fp16_to_bf16(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert float16 tensor back to bfloat16."""
        if tensor.dtype == torch.float16:
            return tensor.float().to(torch.bfloat16)
        elif tensor.dtype == torch.float32:
            return tensor.to(torch.bfloat16)
        return tensor
    
    def _prepare_inputs_for_trt(
        self, 
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for TensorRT execution (bf16 -> fp16)."""
        return (
            self._convert_bf16_to_fp16(hidden_states),
            self._convert_bf16_to_fp16(encoder_hidden_states),
            timestep.float() if timestep.dtype != torch.float32 else timestep,
        )
    
    def _export_to_onnx(
        self, 
        output_path: str, 
        sample_inputs: Dict[str, torch.Tensor]
    ) -> bool:
        """Export PyTorch module to ONNX format."""
        try:
            # Prepare module for export - need to handle bfloat16
            module = self.pytorch_module
            
            # Create wrapper that handles bf16 conversion internally
            class FP16ExportWrapper(nn.Module):
                def __init__(self, orig_module):
                    super().__init__()
                    self.module = orig_module
                
                def forward(self, hidden_states, encoder_hidden_states, timestep):
                    # Convert FP16 inputs to bfloat16 for the actual model
                    hs_bf16 = hidden_states.float().to(torch.bfloat16)
                    enc_bf16 = encoder_hidden_states.float().to(torch.bfloat16)
                    
                    # Run model
                    output = self.module(
                        hidden_states=hs_bf16,
                        encoder_hidden_states=enc_bf16,
                        timestep=timestep,
                    )
                    
                    # Handle different output formats
                    if hasattr(output, 'sample'):
                        output = output.sample
                    elif isinstance(output, tuple):
                        output = output[0]
                    
                    # Convert output back to FP16
                    return output.float().half()
            
            export_wrapper = FP16ExportWrapper(module)
            export_wrapper.eval()
            
            # Prepare sample inputs
            hidden_states = sample_inputs["hidden_states"].to(self.device)
            encoder_hidden_states = sample_inputs["encoder_hidden_states"].to(self.device)
            timestep = sample_inputs["timestep"].to(self.device)
            
            # Dynamic axes for batch and spatial dimensions
            dynamic_axes = {
                "hidden_states": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch"},
                "timestep": {0: "batch"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }
            
            torch.onnx.export(
                export_wrapper,
                (hidden_states, encoder_hidden_states, timestep),
                output_path,
                input_names=["hidden_states", "encoder_hidden_states", "timestep"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )
            
            logger.info(f"ONNX export successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def _build_trt_engine(self, onnx_path: str) -> bool:
        """Build TensorRT engine from ONNX with dynamic shape profiles."""
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(i)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
            
            # Enable FP16 mode
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TensorRT FP16 mode enabled")
            
            # Create optimization profile with dynamic shapes
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
            
            # Deserialize engine
            runtime = trt.Runtime(TRT_LOGGER)
            self._trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            self._trt_context = self._trt_engine.create_execution_context()
            
            logger.info("TensorRT engine built successfully")
            return True
            
        except ImportError:
            logger.error("TensorRT not installed")
            return False
        except Exception as e:
            logger.error(f"TensorRT engine build failed: {e}")
            return False
    
    def _trt_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Execute TensorRT inference with dtype conversion."""
        import tensorrt as trt
        
        # Convert inputs from bfloat16 to float16
        hs_fp16, enc_fp16, ts_fp32 = self._prepare_inputs_for_trt(
            hidden_states, encoder_hidden_states, timestep
        )
        
        # Ensure contiguous
        hs_fp16 = hs_fp16.contiguous()
        enc_fp16 = enc_fp16.contiguous()
        ts_fp32 = ts_fp32.contiguous()
        
        # Get output shape (same as hidden_states)
        output_shape = hs_fp16.shape
        output_fp16 = torch.empty(output_shape, dtype=torch.float16, device=self.device)
        
        # Set input shapes for dynamic dimensions
        self._trt_context.set_input_shape("hidden_states", tuple(hs_fp16.shape))
        self._trt_context.set_input_shape("encoder_hidden_states", tuple(enc_fp16.shape))
        self._trt_context.set_input_shape("timestep", tuple(ts_fp32.shape))
        
        # Set tensor addresses
        self._trt_context.set_tensor_address("hidden_states", hs_fp16.data_ptr())
        self._trt_context.set_tensor_address("encoder_hidden_states", enc_fp16.data_ptr())
        self._trt_context.set_tensor_address("timestep", ts_fp32.data_ptr())
        self._trt_context.set_tensor_address("output", output_fp16.data_ptr())
        
        # Execute
        stream = torch.cuda.current_stream().cuda_stream
        success = self._trt_context.execute_async_v3(stream)
        
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # Convert output back to bfloat16
        output_bf16 = self._convert_fp16_to_bf16(output_fp16)
        
        return {"output": output_bf16}
    
    def _pytorch_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Execute PyTorch inference (fallback)."""
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = self.pytorch_module(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
        
        # Handle different output formats
        if hasattr(output, 'sample'):
            output = output.sample
        elif isinstance(output, tuple):
            output = output[0]
        
        return {"output": output}
    
    def _should_validate_quality(self) -> bool:
        """Check if we should run quality validation on this call."""
        self._validation_call_count += 1
        return (
            self._validation_call_count % self.validation_frequency == 0
            or self._validation_call_count <= 5  # Always validate first few
        )
    
    def _validate_quality(
        self,
        trt_output: torch.Tensor,
        pytorch_output: torch.Tensor,
    ) -> bool:
        """Validate TRT output quality against PyTorch reference.
        
        Returns True if quality is acceptable, False otherwise.
        """
        result = self._quality_validator.validate_bf16_to_fp16(
            trt_output, pytorch_output
        )
        
        self._last_quality_metrics = result.metrics
        
        if not result.valid:
            self._quality_failures += 1
            logger.warning(f"Quality validation failed: {result.reason}")
            
            # If too many failures, mark quality as degraded
            if self._quality_failures >= 3:
                logger.error(
                    "Multiple quality failures detected - switching to PyTorch fallback"
                )
                self._quality_degraded = True
            
            return False
        
        return True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with automatic TRT/PyTorch selection and quality validation.
        
        Args:
            hidden_states: Latent tensor [batch, 4, H/8, W/8], typically bfloat16
            encoder_hidden_states: T5 embeddings [batch, 512, 768]
            timestep: Continuous timestep for flow matching (0.0 to 1.0)
            
        Returns:
            Transformed hidden states in same dtype as input
        """
        # Check if quality has degraded too much
        if self._quality_degraded:
            result = self._pytorch_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                **kwargs,
            )
            return result["output"]
        
        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }
        
        use_trt = (
            self.is_compiled
            and self._inputs_compatible(inputs)
            and not self._compilation_failed
        )
        
        if use_trt:
            try:
                import time
                start = time.time()
                
                result = self._trt_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    **kwargs,
                )
                
                torch.cuda.synchronize()
                
                trt_output = result["output"]
                
                # Validate output
                if not self._validate_output(trt_output):
                    raise RuntimeError("TRT output validation failed (NaN/Inf)")
                
                # Periodic quality validation against PyTorch
                if self._should_validate_quality():
                    with torch.no_grad():
                        pytorch_result = self._pytorch_forward(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timestep,
                        )
                        if not self._validate_quality(trt_output, pytorch_result["output"]):
                            # Quality failed but continue for this call
                            logger.warning("Quality validation failed, will monitor")
                
                self._trt_call_count += 1
                self._trt_total_time += time.time() - start
                
                return trt_output
                
            except Exception as e:
                if self.fallback_enabled:
                    logger.warning(f"TRT execution failed, falling back to PyTorch: {e}")
                    self._fallback_count += 1
                else:
                    raise
        
        # PyTorch fallback
        import time
        start = time.time()
        result = self._pytorch_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **kwargs,
        )
        self._pytorch_call_count += 1
        self._pytorch_total_time += time.time() - start
        
        return result["output"]
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality monitoring statistics."""
        return {
            "validation_count": self._validation_call_count,
            "quality_failures": self._quality_failures,
            "quality_degraded": self._quality_degraded,
            "last_metrics": self._last_quality_metrics,
        }
    
    def reset_quality_stats(self) -> None:
        """Reset quality statistics and re-enable TRT if degraded."""
        self._validation_call_count = 0
        self._quality_failures = 0
        self._quality_degraded = False
        self._last_quality_metrics = None
        logger.info("Quality stats reset, TRT re-enabled")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get extended performance statistics including quality metrics."""
        base_stats = super().get_performance_stats()
        base_stats.update(self.get_quality_stats())
        return base_stats


def create_zimage_trt_wrapper(
    transformer_module: nn.Module,
    quality_threshold: float = 0.05,
    validation_frequency: int = 100,
) -> TRTZImageTransformerWrapper:
    """Factory function to create Z-Image TensorRT wrapper.
    
    Args:
        transformer_module: The Z-Image transformer module to wrap
        quality_threshold: Maximum acceptable relative error (default 5%)
        validation_frequency: How often to validate quality (default every 100 calls)
        
    Returns:
        Configured TRTZImageTransformerWrapper
    """
    wrapper = TRTZImageTransformerWrapper(
        pytorch_module=transformer_module,
        quality_threshold=quality_threshold,
        validation_frequency=validation_frequency,
    )
    
    logger.info(
        f"Created Z-Image TRT wrapper with quality_threshold={quality_threshold}, "
        f"validation_frequency={validation_frequency}"
    )
    
    return wrapper
