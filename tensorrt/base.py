"""Base TensorRT wrapper with automatic PyTorch fallback."""
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TensorRTModuleWrapper(ABC, nn.Module):
    """Abstract base class for TensorRT-compiled module wrappers.
    
    Provides:
    - Automatic fallback to PyTorch on TRT failure
    - Input/output tensor management
    - CUDA stream handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        pytorch_module: nn.Module,
        fallback_enabled: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.pytorch_module = pytorch_module
        self.fallback_enabled = fallback_enabled
        self.device = device
        
        # TensorRT state
        self._trt_engine = None
        self._trt_context = None
        self._compiled = False
        self._compilation_failed = False
        
        # I/O buffer management
        self._input_buffers: Dict[str, torch.Tensor] = {}
        self._output_buffers: Dict[str, torch.Tensor] = {}
        
        # Performance tracking
        self._trt_call_count = 0
        self._pytorch_call_count = 0
        self._trt_total_time = 0.0
        self._pytorch_total_time = 0.0
        self._fallback_count = 0
        
        # CUDA stream for TRT execution
        self._trt_stream: Optional[torch.cuda.Stream] = None
    
    @property
    def is_compiled(self) -> bool:
        """Check if TRT engine is compiled and ready."""
        return self._compiled and not self._compilation_failed
    
    @abstractmethod
    def get_input_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return input tensor specifications.
        
        Returns:
            Dict mapping input names to specs with keys:
            - dtype: torch dtype
            - min_shape: minimum shape tuple
            - opt_shape: optimal shape tuple
            - max_shape: maximum shape tuple
        """
        pass
    
    @abstractmethod
    def get_output_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return output tensor specifications."""
        pass
    
    @abstractmethod
    def _export_to_onnx(self, output_path: str, sample_inputs: Dict[str, torch.Tensor]) -> bool:
        """Export PyTorch module to ONNX format."""
        pass
    
    @abstractmethod
    def _build_trt_engine(self, onnx_path: str) -> bool:
        """Build TensorRT engine from ONNX."""
        pass
    
    def compile(self, force: bool = False) -> bool:
        """Compile PyTorch module to TensorRT.
        
        Args:
            force: If True, recompile even if already compiled
            
        Returns:
            True if compilation succeeded
        """
        if self._compiled and not force:
            logger.info("TensorRT engine already compiled")
            return True
        
        if self._compilation_failed and not force:
            logger.warning("Previous compilation failed, use force=True to retry")
            return False
        
        logger.info(f"Compiling {self.__class__.__name__} to TensorRT...")
        start_time = time.time()
        
        try:
            # Generate sample inputs
            sample_inputs = self._create_sample_inputs()
            
            # Export to ONNX
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                onnx_path = f.name
            
            if not self._export_to_onnx(onnx_path, sample_inputs):
                raise RuntimeError("ONNX export failed")
            
            # Build TRT engine
            if not self._build_trt_engine(onnx_path):
                raise RuntimeError("TensorRT engine build failed")
            
            # Allocate I/O buffers
            self._allocate_buffers()
            
            # Create CUDA stream
            self._trt_stream = torch.cuda.Stream()
            
            self._compiled = True
            self._compilation_failed = False
            
            elapsed = time.time() - start_time
            logger.info(f"TensorRT compilation succeeded in {elapsed:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            self._compilation_failed = True
            self._compiled = False
            return False
    
    def _create_sample_inputs(self) -> Dict[str, torch.Tensor]:
        """Create sample inputs for ONNX export."""
        sample_inputs = {}
        for name, spec in self.get_input_spec().items():
            shape = spec.get("opt_shape", spec.get("min_shape"))
            dtype = spec.get("dtype", torch.float32)
            sample_inputs[name] = torch.randn(shape, dtype=dtype, device=self.device)
        return sample_inputs
    
    def _allocate_buffers(self) -> None:
        """Allocate I/O buffers for TRT execution."""
        for name, spec in self.get_input_spec().items():
            shape = spec.get("max_shape", spec.get("opt_shape"))
            dtype = spec.get("dtype", torch.float32)
            self._input_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
        
        for name, spec in self.get_output_spec().items():
            shape = spec.get("max_shape", spec.get("opt_shape"))
            dtype = spec.get("dtype", torch.float32)
            self._output_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
    
    def _validate_output(self, output: torch.Tensor) -> bool:
        """Validate TRT output for NaN/Inf."""
        if torch.isnan(output).any():
            logger.warning("TensorRT output contains NaN values")
            return False
        if torch.isinf(output).any():
            logger.warning("TensorRT output contains Inf values")
            return False
        return True
    
    @abstractmethod
    def _trt_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Execute TensorRT inference."""
        pass
    
    def _pytorch_forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Execute PyTorch inference (fallback)."""
        return self.pytorch_module(**inputs)
    
    def forward(self, **inputs) -> Any:
        """Forward pass with automatic TRT/PyTorch selection.
        
        Uses TensorRT if compiled and inputs are compatible,
        otherwise falls back to PyTorch.
        """
        use_trt = (
            self.is_compiled
            and self._inputs_compatible(inputs)
            and not self._compilation_failed
        )
        
        if use_trt:
            try:
                start = time.time()
                
                with torch.cuda.stream(self._trt_stream):
                    output = self._trt_forward(**inputs)
                
                # Synchronize and validate
                self._trt_stream.synchronize()
                
                # Validate output
                output_tensor = output if isinstance(output, torch.Tensor) else list(output.values())[0]
                if not self._validate_output(output_tensor):
                    raise RuntimeError("TRT output validation failed")
                
                self._trt_call_count += 1
                self._trt_total_time += time.time() - start
                return output
                
            except Exception as e:
                if self.fallback_enabled:
                    logger.warning(f"TRT execution failed, falling back to PyTorch: {e}")
                    self._fallback_count += 1
                else:
                    raise
        
        # PyTorch fallback
        start = time.time()
        output = self._pytorch_forward(**inputs)
        self._pytorch_call_count += 1
        self._pytorch_total_time += time.time() - start
        return output
    
    def _inputs_compatible(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """Check if inputs are compatible with compiled TRT engine."""
        input_spec = self.get_input_spec()
        
        for name, spec in input_spec.items():
            if name not in inputs:
                return False
            
            tensor = inputs[name]
            max_shape = spec.get("max_shape")
            min_shape = spec.get("min_shape")
            
            if max_shape and len(tensor.shape) == len(max_shape):
                for i, (t_dim, max_dim) in enumerate(zip(tensor.shape, max_shape)):
                    if t_dim > max_dim:
                        logger.debug(f"Input {name} dim {i} ({t_dim}) exceeds max ({max_dim})")
                        return False
            
            if min_shape and len(tensor.shape) == len(min_shape):
                for i, (t_dim, min_dim) in enumerate(zip(tensor.shape, min_shape)):
                    if t_dim < min_dim:
                        logger.debug(f"Input {name} dim {i} ({t_dim}) below min ({min_dim})")
                        return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        trt_avg = self._trt_total_time / max(1, self._trt_call_count)
        pytorch_avg = self._pytorch_total_time / max(1, self._pytorch_call_count)
        
        return {
            "trt_calls": self._trt_call_count,
            "pytorch_calls": self._pytorch_call_count,
            "fallback_count": self._fallback_count,
            "trt_avg_ms": trt_avg * 1000,
            "pytorch_avg_ms": pytorch_avg * 1000,
            "speedup": pytorch_avg / trt_avg if trt_avg > 0 else 0,
            "is_compiled": self.is_compiled,
            "compilation_failed": self._compilation_failed,
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._trt_call_count = 0
        self._pytorch_call_count = 0
        self._trt_total_time = 0.0
        self._pytorch_total_time = 0.0
        self._fallback_count = 0
    
    def cleanup(self) -> None:
        """Release TensorRT resources."""
        self._trt_engine = None
        self._trt_context = None
        self._input_buffers.clear()
        self._output_buffers.clear()
        self._compiled = False
        
        if self._trt_stream:
            self._trt_stream = None
        
        torch.cuda.empty_cache()
        logger.info(f"Cleaned up TensorRT resources for {self.__class__.__name__}")


class SafeTensorRTExecution:
    """Context manager for safe TensorRT execution with fallback."""
    
    def __init__(
        self,
        trt_wrapper: TensorRTModuleWrapper,
        pytorch_fallback: nn.Module,
    ):
        self.trt_wrapper = trt_wrapper
        self.pytorch_fallback = pytorch_fallback
    
    def __call__(self, **inputs) -> Any:
        """Execute with automatic fallback handling."""
        try:
            if self.trt_wrapper.is_compiled:
                return self.trt_wrapper(**inputs)
        except Exception as e:
            logger.warning(f"TRT execution failed: {e}")
        
        return self.pytorch_fallback(**inputs)
