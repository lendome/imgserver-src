"""Graceful TensorRT fallback mechanism with comprehensive error handling."""
import logging
import time
import functools
from typing import Callable, Any, Optional, TypeVar, Dict
from dataclasses import dataclass, field
from enum import Enum
import torch

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    """Reasons for falling back to PyTorch."""
    TRT_NOT_COMPILED = "tensorrt_not_compiled"
    TRT_EXECUTION_ERROR = "tensorrt_execution_error"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    INPUT_SHAPE_MISMATCH = "input_shape_mismatch"
    CUDA_OOM = "cuda_out_of_memory"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class FallbackStats:
    """Statistics for fallback tracking."""
    total_calls: int = 0
    trt_successes: int = 0
    pytorch_fallbacks: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)
    last_fallback_reason: Optional[str] = None
    last_fallback_time: Optional[float] = None
    consecutive_fallbacks: int = 0
    
    def record_success(self):
        self.total_calls += 1
        self.trt_successes += 1
        self.consecutive_fallbacks = 0
    
    def record_fallback(self, reason: FallbackReason):
        self.total_calls += 1
        self.pytorch_fallbacks += 1
        self.consecutive_fallbacks += 1
        self.last_fallback_reason = reason.value
        self.last_fallback_time = time.time()
        self.fallback_reasons[reason.value] = self.fallback_reasons.get(reason.value, 0) + 1

class TensorRTFallbackManager:
    """Manages TensorRT fallback logic with adaptive behavior."""
    
    def __init__(
        self,
        max_consecutive_fallbacks: int = 5,
        disable_trt_after_failures: bool = True,
        cooldown_seconds: float = 60.0,
    ):
        self.max_consecutive_fallbacks = max_consecutive_fallbacks
        self.disable_trt_after_failures = disable_trt_after_failures
        self.cooldown_seconds = cooldown_seconds
        
        self._stats = FallbackStats()
        self._trt_disabled = False
        self._disabled_time: Optional[float] = None
    
    @property
    def stats(self) -> FallbackStats:
        return self._stats
    
    @property
    def trt_disabled(self) -> bool:
        # Check if cooldown has passed
        if self._trt_disabled and self._disabled_time:
            if time.time() - self._disabled_time > self.cooldown_seconds:
                logger.info("TRT cooldown expired, re-enabling")
                self._trt_disabled = False
                self._disabled_time = None
        return self._trt_disabled
    
    def should_use_trt(self) -> bool:
        """Determine if TensorRT should be attempted."""
        if self.trt_disabled:
            return False
        return True
    
    def record_success(self):
        """Record successful TensorRT execution."""
        self._stats.record_success()
    
    def record_fallback(self, reason: FallbackReason, error: Optional[Exception] = None):
        """Record fallback event."""
        self._stats.record_fallback(reason)
        
        if error:
            logger.warning(f"TRT fallback ({reason.value}): {error}")
        else:
            logger.warning(f"TRT fallback: {reason.value}")
        
        # Check if we should disable TRT
        if (self.disable_trt_after_failures and 
            self._stats.consecutive_fallbacks >= self.max_consecutive_fallbacks):
            logger.error(f"Disabling TRT after {self.max_consecutive_fallbacks} consecutive failures")
            self._trt_disabled = True
            self._disabled_time = time.time()
    
    def reset(self):
        """Reset fallback state."""
        self._stats = FallbackStats()
        self._trt_disabled = False
        self._disabled_time = None

# Global fallback manager instance
_fallback_manager: Optional[TensorRTFallbackManager] = None

def get_fallback_manager() -> TensorRTFallbackManager:
    """Get global fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = TensorRTFallbackManager()
    return _fallback_manager

def with_trt_fallback(pytorch_fn: Callable) -> Callable:
    """Decorator for TensorRT functions with automatic PyTorch fallback.
    
    Usage:
        @with_trt_fallback
        def trt_forward(self, x):
            return self._trt_engine(x)
    """
    @functools.wraps(pytorch_fn)
    def wrapper(self, *args, **kwargs):
        manager = get_fallback_manager()
        
        # Check if we should attempt TRT
        if not manager.should_use_trt():
            return self._pytorch_forward(*args, **kwargs)
        
        if not getattr(self, 'is_compiled', False):
            manager.record_fallback(FallbackReason.TRT_NOT_COMPILED)
            return self._pytorch_forward(*args, **kwargs)
        
        try:
            result = pytorch_fn(self, *args, **kwargs)
            manager.record_success()
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            manager.record_fallback(FallbackReason.CUDA_OOM, e)
            torch.cuda.empty_cache()
            return self._pytorch_forward(*args, **kwargs)
            
        except Exception as e:
            manager.record_fallback(FallbackReason.TRT_EXECUTION_ERROR, e)
            return self._pytorch_forward(*args, **kwargs)
    
    return wrapper

def safe_trt_execute(
    trt_callable: Callable,
    pytorch_fallback: Callable,
    *args,
    validate_output: bool = True,
    **kwargs
) -> Any:
    """Execute TRT with fallback, standalone function version."""
    manager = get_fallback_manager()
    
    if not manager.should_use_trt():
        return pytorch_fallback(*args, **kwargs)
    
    try:
        result = trt_callable(*args, **kwargs)
        
        if validate_output and isinstance(result, torch.Tensor):
            if torch.isnan(result).any() or torch.isinf(result).any():
                manager.record_fallback(FallbackReason.OUTPUT_VALIDATION_FAILED)
                return pytorch_fallback(*args, **kwargs)
        
        manager.record_success()
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        manager.record_fallback(FallbackReason.CUDA_OOM, e)
        torch.cuda.empty_cache()
        return pytorch_fallback(*args, **kwargs)
        
    except Exception as e:
        manager.record_fallback(FallbackReason.TRT_EXECUTION_ERROR, e)
        return pytorch_fallback(*args, **kwargs)
