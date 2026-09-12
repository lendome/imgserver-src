"""RAM Manager - System RAM tracking for CPU model parking."""

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 8GB reserved for system
SYSTEM_RESERVE_BYTES = 8 * 1024 * 1024 * 1024


class RAMManager:
    """Singleton RAM manager for tracking system memory."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_total_ram(self) -> int:
        """Get total system RAM in bytes."""
        if not PSUTIL_AVAILABLE:
            return 64 * 1024 * 1024 * 1024  # 64GB fake
        return psutil.virtual_memory().total
    
    def get_free_ram(self) -> int:
        """Get available RAM in bytes."""
        if not PSUTIL_AVAILABLE:
            return 32 * 1024 * 1024 * 1024  # 32GB fake
        return psutil.virtual_memory().available
    
    def get_used_ram(self) -> int:
        """Get used RAM in bytes."""
        if not PSUTIL_AVAILABLE:
            return 32 * 1024 * 1024 * 1024  # 32GB fake
        return psutil.virtual_memory().used
    
    def get_safe_parking_space(self) -> int:
        """Get RAM available for model parking (leaves 8GB for system)."""
        free = self.get_free_ram()
        safe = free - SYSTEM_RESERVE_BYTES
        return max(0, safe)
    
    def can_park_model(self, model_size_bytes: int) -> bool:
        """Check if we have enough RAM to park a model to CPU."""
        return self.get_safe_parking_space() >= model_size_bytes


# Global singleton instance
ram_manager = RAMManager()
