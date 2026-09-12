"""GPU lock for exclusive access during generation."""
import threading
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VideoGenerationActiveError(Exception):
    """Raised when attempting to acquire lock while video generation is active."""
    pass

class ClientDisconnectedWhileWaiting(Exception):
    """Raised when client disconnects while waiting for GPU lock."""
    pass

class GPULock:
    """Ensures exclusive GPU access for generation."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance._holder = None
            cls._instance._video_active = False
            cls._instance._video_lock = threading.Lock()
        return cls._instance
    
    @contextmanager
    def acquire(self, holder: str = "unknown"):
        """Context manager for GPU lock with disconnect detection."""
        acquired = self._lock.acquire(timeout=0.1)
        if not acquired:
            logger.info(f"GPU busy, {holder} waiting...")
            while not self._lock.acquire(timeout=0.5):
                try:
                    from .abort import is_client_disconnected
                    if is_client_disconnected():
                        logger.info(f"Client disconnected while {holder} was waiting for GPU lock")
                        raise ClientDisconnectedWhileWaiting(f"Client disconnected while waiting for GPU lock")
                except ImportError:
                    pass
        
        self._holder = holder
        logger.debug(f"GPU lock acquired by {holder}")
        try:
            yield
        finally:
            self._holder = None
            self._lock.release()
            logger.debug(f"GPU lock released by {holder}")
    
    def is_busy(self) -> bool:
        """Check if GPU is currently in use."""
        return self._lock.locked()
    
    @property
    def current_holder(self) -> str:
        return self._holder
    
    def set_video_active(self, active: bool) -> None:
        """Set the video generation active flag."""
        with self._video_lock:
            self._video_active = active
            logger.debug(f"Video generation active: {active}")
    
    def is_video_active(self) -> bool:
        """Check if video generation is currently running."""
        with self._video_lock:
            return self._video_active

gpu_lock = GPULock()
