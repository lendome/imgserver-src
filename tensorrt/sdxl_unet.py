"""TensorRT wrapper for SDXL UNet with dynamic shape support."""
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import TensorRTModuleWrapper
from .cache_manager import get_engine_cache
from .config import TensorRTSettings, validate_precision

logger = logging.getLogger(__name__)


class TRTSDXLUNetWrapper(TensorRTModuleWrapper):
    """TensorRT wrapper for SDXL UNet with FP16 precision and dynamic shapes.
    
    Supports:
    - Dynamic batch sizes (1-1)
    - Dynamic resolutions (512x512 to 1536x1536)
    - SDXL-specific conditioning (text_embeds, time_ids)
    - Automatic caching and cache invalidation
    """
    
    # SDXL-specific constants
    LATENT_CHANNELS = 4
    TEXT_EMBED_DIM = 2048
    POOLED_EMBED_DIM = 1280
    TIME_IDS_DIM = 6
    
    def __init__(
        self,
        pytorch_module: nn.Module,
        settings: Optional[TensorRTSettings] = None,
        fallback_enabled: bool = True,
        device: str = "cuda",
    ):
        """Initialize SDXL UNet TensorRT wrapper.
        
        Args:
            pytorch_module: The PyTorch SDXL UNet module
            settings: TensorRT settings, uses defaults if None
            fallback_enabled: Enable PyTorch fallback on TRT failure
            device: Target device
        """
        super().__init__(pytorch_module, fallback_enabled, device)
        
        self.settings = settings or TensorRTSettings()
        self._cache = get_engine_cache(self.settings.engine_cache_dir)
        self._cache_key: Optional[str] = None
        
        # Resolution profiles (image dimensions -> latent dimensions / 8)
        self._min_res = (self.settings.min_height // 8, self.settings.min_width // 8)
        self._opt_res = (self.settings.opt_height // 8, self.settings.opt_width // 8)
        self._max_res = (self.settings.max_height // 8, self.settings.max_width // 8)
        
        logger.info(
            f"TRTSDXLUNetWrapper initialized with resolution profiles: "
            f"min={self._min_res}, opt={self._opt_res}, max={self._max_res}"
        )
    
    def get_input_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return input tensor specifications for SDXL UNet.
        
        Returns:
            Dict mapping input names to specs with dtype and shape profiles.
        """
        batch_min = self.settings.min_batch_size
        batch_max = self.settings.max_batch_size
        batch_opt = 1
        
        # Sequence length for text embeddings (77 tokens * 2 for SDXL)
        seq_len = 77
        
        return {
            "sample": {
                "dtype": torch.float16 if self.settings.precision == "fp16" else torch.float32,
                "min_shape": (batch_min, self.LATENT_CHANNELS, self._min_res[0], self._min_res[1]),
                "opt_shape": (batch_opt, self.LATENT_CHANNELS, self._opt_res[0], self._opt_res[1]),
                "max_shape": (batch_max, self.LATENT_CHANNELS, self._max_res[0], self._max_res[1]),
            },
            "timestep": {
                "dtype": torch.int64,
                "min_shape": (batch_min,),
                "opt_shape": (batch_opt,),
                "max_shape": (batch_max,),
            },
            "encoder_hidden_states": {
                "dtype": torch.float16 if self.settings.precision == "fp16" else torch.float32,
                "min_shape": (batch_min, seq_len, self.TEXT_EMBED_DIM),
                "opt_shape": (batch_opt, seq_len, self.TEXT_EMBED_DIM),
                "max_shape": (batch_max, seq_len, self.TEXT_EMBED_DIM),
            },
            "text_embeds": {
                "dtype": torch.float16 if self.settings.precision == "fp16" else torch.float32,
                "min_shape": (batch_min, self.POOLED_EMBED_DIM),
                "opt_shape": (batch_opt, self.POOLED_EMBED_DIM),
                "max_shape": (batch_max, self.POOLED_EMBED_DIM),
            },
            "time_ids": {
                "dtype": torch.float16 if self.settings.precision == "fp16" else torch.float32,
                "min_shape": (batch_min, self.TIME_IDS_DIM),
                "opt_shape": (batch_opt, self.TIME_IDS_DIM),
                "max_shape": (batch_max, self.TIME_IDS_DIM),
            },
        }
    
    def get_output_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return output tensor specifications.
        
        Returns:
            Dict with output sample specification.
        """
        batch_max = self.settings.max_batch_size
        
        return {
            "sample": {
                "dtype": torch.float16 if self.settings.precision == "fp16" else torch.float32,
                "min_shape": (1, self.LATENT_CHANNELS, self._min_res[0], self._min_res[1]),
                "opt_shape": (1, self.LATENT_CHANNELS, self._opt_res[0], self._opt_res[1]),
                "max_shape": (batch_max, self.LATENT_CHANNELS, self._max_res[0], self._max_res[1]),
            }
        }
    
    def _export_to_onnx(self, output_path: str, sample_inputs: Dict[str, torch.Tensor]) -> bool:
        """Export SDXL UNet to ONNX format.
        
        Args:
            output_path: Path for the ONNX file
            sample_inputs: Sample input tensors for tracing
            
        Returns:
            True if export succeeded
        """
        logger.info(f"Exporting SDXL UNet to ONNX: {output_path}")
        
        try:
            # Prepare inputs for export
            sample = sample_inputs["sample"]
            timestep = sample_inputs["timestep"]
            encoder_hidden_states = sample_inputs["encoder_hidden_states"]
            text_embeds = sample_inputs["text_embeds"]
            time_ids = sample_inputs["time_ids"]
            
            # SDXL UNet expects added_cond_kwargs
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            
            # Prepare model for export
            self.pytorch_module.eval()
            
            # Dynamic axes for variable shapes
            dynamic_axes = {
                "sample": {0: "batch", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "text_embeds": {0: "batch"},
                "time_ids": {0: "batch"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }
            
            # Export with ONNX
            with torch.no_grad():
                torch.onnx.export(
                    self.pytorch_module,
                    (sample, timestep, encoder_hidden_states, None, None, None, None, added_cond_kwargs),
                    output_path,
                    input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    do_constant_folding=True,
                    export_params=True,
                )
            
            logger.info(f"ONNX export completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def _build_trt_engine(self, onnx_path: str) -> bool:
        """Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            True if engine build succeeded
        """
        try:
            import tensorrt as trt
        except ImportError:
            logger.error("TensorRT not installed. Install with: pip install tensorrt")
            return False
        
        logger.info(f"Building TensorRT engine from {onnx_path}")
        
        try:
            # Create TensorRT builder
            trt_logger = trt.Logger(
                trt.Logger.VERBOSE if self.settings.verbose_logging else trt.Logger.WARNING
            )
            builder = trt.Builder(trt_logger)
            
            # Create network with explicit batch
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            
            # Parse ONNX
            parser = trt.OnnxParser(network, trt_logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(i)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            
            # Set workspace size
            workspace_bytes = int(self.settings.max_workspace_gb * (1 << 30))
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            
            # Set precision
            if self.settings.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 mode enabled")
                else:
                    logger.warning("FP16 not supported on this platform, using FP32")
            elif self.settings.precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("INT8 mode enabled")
                else:
                    logger.warning("INT8 not supported on this platform")
            
            # Set optimization level
            config.builder_optimization_level = self.settings.builder_optimization_level
            
            # Configure dynamic shapes
            if self.settings.dynamic_shapes:
                profile = builder.create_optimization_profile()
                input_spec = self.get_input_spec()
                
                for name, spec in input_spec.items():
                    profile.set_shape(
                        name,
                        spec["min_shape"],
                        spec["opt_shape"],
                        spec["max_shape"],
                    )
                    logger.debug(
                        f"Shape profile for {name}: "
                        f"min={spec['min_shape']}, opt={spec['opt_shape']}, max={spec['max_shape']}"
                    )
                
                config.add_optimization_profile(profile)
            
            # Load timing cache if available
            timing_cache_path = self._cache.get_timing_cache_path()
            timing_cache = None
            
            if self.settings.timing_cache_enabled and timing_cache_path.exists():
                try:
                    with open(timing_cache_path, 'rb') as f:
                        timing_cache = config.create_timing_cache(f.read())
                        config.set_timing_cache(timing_cache, ignore_mismatch=True)
                        logger.info("Loaded timing cache")
                except Exception as e:
                    logger.warning(f"Failed to load timing cache: {e}")
            else:
                timing_cache = config.create_timing_cache(b"")
                config.set_timing_cache(timing_cache, ignore_mismatch=True)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("TensorRT engine build failed")
                return False
            
            # Save timing cache
            if self.settings.timing_cache_enabled and timing_cache is not None:
                try:
                    timing_cache = config.get_timing_cache()
                    with open(timing_cache_path, 'wb') as f:
                        f.write(timing_cache.serialize())
                    logger.info("Saved timing cache")
                except Exception as e:
                    logger.warning(f"Failed to save timing cache: {e}")
            
            # Deserialize and store engine
            runtime = trt.Runtime(trt_logger)
            self._trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            if self._trt_engine is None:
                logger.error("Failed to deserialize TensorRT engine")
                return False
            
            # Create execution context
            self._trt_context = self._trt_engine.create_execution_context()
            
            # Cache the engine
            self._cache_engine(serialized_engine)
            
            logger.info("TensorRT engine built successfully")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT engine build failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cache_engine(self, serialized_engine: bytes) -> None:
        """Cache the built engine for future use.
        
        Args:
            serialized_engine: Serialized TensorRT engine bytes
        """
        try:
            model_config = {
                "type": "sdxl_unet",
                "latent_channels": self.LATENT_CHANNELS,
                "text_embed_dim": self.TEXT_EMBED_DIM,
            }
            
            tensorrt_config = {
                "precision": self.settings.precision,
                "max_workspace_gb": self.settings.max_workspace_gb,
                "builder_optimization_level": self.settings.builder_optimization_level,
            }
            
            input_shapes = {
                name: list(spec["opt_shape"])
                for name, spec in self.get_input_spec().items()
            }
            
            self._cache_key = self._cache.compute_cache_key(
                "sdxl_unet", model_config, tensorrt_config
            )
            
            self._cache.save_engine(
                self._cache_key,
                serialized_engine,
                model_config,
                tensorrt_config,
                input_shapes,
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache engine: {e}")
    
    def try_load_cached_engine(self) -> bool:
        """Attempt to load a cached engine.
        
        Returns:
            True if a valid cached engine was loaded
        """
        try:
            import tensorrt as trt
        except ImportError:
            return False
        
        try:
            model_config = {
                "type": "sdxl_unet",
                "latent_channels": self.LATENT_CHANNELS,
                "text_embed_dim": self.TEXT_EMBED_DIM,
            }
            
            tensorrt_config = {
                "precision": self.settings.precision,
                "max_workspace_gb": self.settings.max_workspace_gb,
                "builder_optimization_level": self.settings.builder_optimization_level,
            }
            
            self._cache_key = self._cache.compute_cache_key(
                "sdxl_unet", model_config, tensorrt_config
            )
            
            engine_bytes = self._cache.load_engine(self._cache_key)
            
            if engine_bytes is None:
                return False
            
            # Deserialize engine
            trt_logger = trt.Logger(
                trt.Logger.VERBOSE if self.settings.verbose_logging else trt.Logger.WARNING
            )
            runtime = trt.Runtime(trt_logger)
            self._trt_engine = runtime.deserialize_cuda_engine(engine_bytes)
            
            if self._trt_engine is None:
                logger.warning("Failed to deserialize cached engine")
                return False
            
            self._trt_context = self._trt_engine.create_execution_context()
            self._allocate_buffers()
            self._trt_stream = torch.cuda.Stream()
            self._compiled = True
            
            logger.info("Loaded cached TensorRT engine")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached engine: {e}")
            return False
    
    def compile(self, force: bool = False) -> bool:
        """Compile SDXL UNet to TensorRT.
        
        First attempts to load from cache, then builds if necessary.
        
        Args:
            force: If True, rebuild even if cached engine exists
            
        Returns:
            True if compilation/loading succeeded
        """
        if self._compiled and not force:
            logger.info("TensorRT engine already compiled")
            return True
        
        # Try loading from cache first
        if not force and self.try_load_cached_engine():
            return True
        
        # Build new engine
        return super().compile(force=force)
    
    def _trt_forward(self, **inputs) -> torch.Tensor:
        """Execute TensorRT inference for SDXL UNet.
        
        Args:
            **inputs: Input tensors including sample, timestep, 
                     encoder_hidden_states, and added_cond_kwargs
                     
        Returns:
            UNet output sample tensor
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("TensorRT not installed")
        
        # Extract inputs
        sample = inputs.get("sample")
        timestep = inputs.get("timestep")
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        
        # Handle SDXL-specific kwargs
        added_cond_kwargs = inputs.get("added_cond_kwargs", {})
        text_embeds = added_cond_kwargs.get("text_embeds") or inputs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids") or inputs.get("time_ids")
        
        if sample is None:
            raise ValueError("Missing required input: sample")
        if timestep is None:
            raise ValueError("Missing required input: timestep")
        if encoder_hidden_states is None:
            raise ValueError("Missing required input: encoder_hidden_states")
        if text_embeds is None:
            raise ValueError("Missing required input: text_embeds (in added_cond_kwargs)")
        if time_ids is None:
            raise ValueError("Missing required input: time_ids (in added_cond_kwargs)")
        
        # Convert to correct dtype
        dtype = torch.float16 if self.settings.precision == "fp16" else torch.float32
        sample = sample.to(dtype=dtype, device=self.device).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(dtype=dtype, device=self.device).contiguous()
        text_embeds = text_embeds.to(dtype=dtype, device=self.device).contiguous()
        time_ids = time_ids.to(dtype=dtype, device=self.device).contiguous()
        
        # Timestep should be int64
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], dtype=torch.int64, device=self.device)
        timestep = timestep.to(dtype=torch.int64, device=self.device).contiguous()
        
        # Ensure batch dimension for timestep
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # Set input shapes for dynamic profiles
        batch_size = sample.shape[0]
        height = sample.shape[2]
        width = sample.shape[3]
        
        # Set input tensor addresses and shapes
        self._trt_context.set_input_shape("sample", sample.shape)
        self._trt_context.set_input_shape("timestep", timestep.shape)
        self._trt_context.set_input_shape("encoder_hidden_states", encoder_hidden_states.shape)
        self._trt_context.set_input_shape("text_embeds", text_embeds.shape)
        self._trt_context.set_input_shape("time_ids", time_ids.shape)
        
        # Allocate output tensor
        output_shape = (batch_size, self.LATENT_CHANNELS, height, width)
        output = torch.empty(output_shape, dtype=dtype, device=self.device)
        
        # Set tensor addresses
        self._trt_context.set_tensor_address("sample", sample.data_ptr())
        self._trt_context.set_tensor_address("timestep", timestep.data_ptr())
        self._trt_context.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
        self._trt_context.set_tensor_address("text_embeds", text_embeds.data_ptr())
        self._trt_context.set_tensor_address("time_ids", time_ids.data_ptr())
        self._trt_context.set_tensor_address("output", output.data_ptr())
        
        # Execute inference
        success = self._trt_context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output
    
    def _pytorch_forward(self, **inputs) -> torch.Tensor:
        """Execute PyTorch inference (fallback).
        
        Args:
            **inputs: Input tensors
            
        Returns:
            UNet output sample tensor
        """
        sample = inputs.get("sample")
        timestep = inputs.get("timestep")
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        
        # Handle SDXL-specific kwargs
        added_cond_kwargs = inputs.get("added_cond_kwargs")
        
        if added_cond_kwargs is None:
            text_embeds = inputs.get("text_embeds")
            time_ids = inputs.get("time_ids")
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
        
        with torch.no_grad():
            output = self.pytorch_module(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        
        # Handle tuple output from UNet
        if isinstance(output, tuple):
            return output[0]
        return output.sample if hasattr(output, 'sample') else output
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with automatic TRT/PyTorch selection.
        
        Args:
            sample: Latent tensor [batch, 4, height/8, width/8]
            timestep: Timestep value
            encoder_hidden_states: Text embeddings [batch, seq_len, 2048]
            added_cond_kwargs: Dict with text_embeds and time_ids
            **kwargs: Additional arguments (ignored for TRT path)
            
        Returns:
            UNet output sample tensor
        """
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": added_cond_kwargs,
        }
        
        # Extract from added_cond_kwargs for TRT path
        if added_cond_kwargs:
            inputs["text_embeds"] = added_cond_kwargs.get("text_embeds")
            inputs["time_ids"] = added_cond_kwargs.get("time_ids")
        
        return super().forward(**inputs)
