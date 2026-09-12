"""VRAM Manager - Aggressive VRAM management for stable diffusion."""

import gc
import logging
from typing import Callable

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class VRAMManager:
    """Singleton VRAM manager with aggressive memory management."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_total(self) -> int:
        """Get total VRAM in bytes."""
        if not CUDA_AVAILABLE:
            return 64 * 1024 * 1024 * 1024
        _, total = torch.cuda.mem_get_info()
        return total
    
    def get_free(self) -> int:
        """Get actual free VRAM in bytes."""
        if not CUDA_AVAILABLE:
            return 32 * 1024 * 1024 * 1024
        free, _ = torch.cuda.mem_get_info()
        return free
    
    def get_used(self) -> int:
        """Get actual used VRAM in bytes."""
        return self.get_total() - self.get_free()
    
    def get_allocated(self) -> int:
        """Get PyTorch allocated memory."""
        if not CUDA_AVAILABLE:
            return 0
        return torch.cuda.memory_allocated()
    
    def get_reserved(self) -> int:
        """Get PyTorch reserved memory."""
        if not CUDA_AVAILABLE:
            return 0
        return torch.cuda.memory_reserved()
    
    def cleanup(self) -> None:
        """Standard cleanup - gc + empty cache."""
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
    
    def aggressive_cleanup(self) -> None:
        """Aggressive cleanup for maximum VRAM recovery."""
        # Multiple GC passes to catch circular references
        for _ in range(3):
            gc.collect()
        
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # IPC collect for Windows
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            
            # Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    
    def nuclear_cleanup(self) -> None:
        """Nuclear option - maximum memory recovery."""
        logger.warning("Running nuclear VRAM cleanup")
        
        # Force Python garbage collection
        for _ in range(5):
            gc.collect()
        
        if CUDA_AVAILABLE:
            # Synchronize all CUDA operations
            torch.cuda.synchronize()
            
            # Empty the cache
            torch.cuda.empty_cache()
            
            # IPC collect
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            
            # Reset stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass
            
            # Force synchronize again
            torch.cuda.synchronize()
        
        # Final GC
        gc.collect()
        
        logger.info(f"After nuclear cleanup: {self.get_free() / 1e9:.2f}GB free")
    
    def can_fit(self, needed_bytes: int) -> bool:
        """Check if we have space for needed_bytes (with 10% safety margin)."""
        free = self.get_free()
        required = int(needed_bytes * 1.1)
        return free >= required
    
    def ensure_free(self, needed_bytes: int, unload_callback: Callable[[], bool]) -> bool:
        """Ensure we have enough free VRAM by unloading if necessary."""
        self.cleanup()
        
        while not self.can_fit(needed_bytes):
            if not unload_callback():
                return False
            self.aggressive_cleanup()
        
        return True
    
    def log_status(self) -> None:
        """Log current VRAM status."""
        if not CUDA_AVAILABLE:
            logger.info("CUDA not available")
            return
        
        total = self.get_total() / 1e9
        free = self.get_free() / 1e9
        used = self.get_used() / 1e9
        allocated = self.get_allocated() / 1e9
        reserved = self.get_reserved() / 1e9
        
        logger.info(
            f"VRAM: {used:.2f}/{total:.2f}GB used, {free:.2f}GB free, "
            f"PyTorch: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


# Global singleton instance
vram_manager = VRAMManager()
