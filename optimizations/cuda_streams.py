"""CUDA Stream management for overlapped execution on RTX 3090 Ti."""
import torch
from typing import Optional, Callable, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class CUDAStreamManager:
    """Manage CUDA streams for overlapped H2D copy and compute operations.
    
    RTX 3090 Ti has high memory bandwidth (936 GB/s) which benefits from
    overlapping memory transfers with compute operations.
    """
    
    _instance: Optional['CUDAStreamManager'] = None
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # Create dedicated streams
        self.compute_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        self.trt_stream = torch.cuda.Stream()
        
        # Default stream reference
        self.default_stream = torch.cuda.current_stream()
        
        # Events for synchronization
        self._copy_done_event = torch.cuda.Event()
        self._compute_done_event = torch.cuda.Event()
        
        logger.info("Initialized CUDA stream manager with 3 streams")
    
    @classmethod
    def get_instance(cls) -> 'CUDAStreamManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @contextmanager
    def compute_context(self):
        """Context manager for compute operations."""
        with torch.cuda.stream(self.compute_stream):
            yield self.compute_stream
    
    @contextmanager
    def copy_context(self):
        """Context manager for H2D/D2H copy operations."""
        with torch.cuda.stream(self.copy_stream):
            yield self.copy_stream
    
    @contextmanager
    def trt_context(self):
        """Context manager for TensorRT operations."""
        with torch.cuda.stream(self.trt_stream):
            yield self.trt_stream
    
    def overlap_copy_and_compute(
        self,
        copy_fn: Callable[[], Any],
        compute_fn: Callable[[], Any],
    ) -> tuple:
        """Execute copy and compute operations with overlap.
        
        Args:
            copy_fn: Function performing memory transfer
            compute_fn: Function performing compute operation
            
        Returns:
            Tuple of (copy_result, compute_result)
        """
        # Start compute on compute stream
        with torch.cuda.stream(self.compute_stream):
            compute_result = compute_fn()
            self._compute_done_event.record(self.compute_stream)
        
        # Start copy on copy stream (overlaps with compute)
        with torch.cuda.stream(self.copy_stream):
            copy_result = copy_fn()
            self._copy_done_event.record(self.copy_stream)
        
        # Wait for both to complete
        self._compute_done_event.synchronize()
        self._copy_done_event.synchronize()
        
        return copy_result, compute_result
    
    def prefetch_to_gpu(self, tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """Prefetch tensor to GPU on copy stream."""
        with torch.cuda.stream(self.copy_stream):
            gpu_tensor = tensor.to('cuda', non_blocking=non_blocking)
            self._copy_done_event.record(self.copy_stream)
        return gpu_tensor
    
    def wait_for_copy(self):
        """Wait for copy stream operations to complete."""
        self._copy_done_event.synchronize()
    
    def wait_for_compute(self):
        """Wait for compute stream operations to complete."""
        self._compute_done_event.synchronize()
    
    def synchronize_all(self):
        """Synchronize all streams."""
        self.compute_stream.synchronize()
        self.copy_stream.synchronize()
        self.trt_stream.synchronize()
        torch.cuda.synchronize()
    
    def get_stream_for_trt(self) -> int:
        """Get TRT stream's CUDA pointer for TensorRT context."""
        return self.trt_stream.cuda_stream


def get_stream_manager() -> CUDAStreamManager:
    """Get the global CUDA stream manager instance."""
    return CUDAStreamManager.get_instance()
