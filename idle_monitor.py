"""Idle monitor for auto-unloading inactive models."""
import threading
import time
import logging

logger = logging.getLogger(__name__)

class IdleMonitor:
    """Monitors pipeline activity and unloads idle models."""
    
    def __init__(self, idle_timeout: float = 1800.0):  # 30 minutes default
        self.idle_timeout = idle_timeout
        self._last_activity = time.time()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def touch(self):
        """Record activity."""
        with self._lock:
            self._last_activity = time.time()
    
    def start(self):
        """Start the idle monitor thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Idle monitor started (timeout: {self.idle_timeout}s)")
    
    def stop(self):
        """Stop the idle monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Check for idle timeout and unload if needed."""
        from .pipelines import unload_all, get_loaded_pipelines
        
        while self._running:
            time.sleep(60)  # Check every minute
            
            with self._lock:
                idle_time = time.time() - self._last_activity
            
            if idle_time > self.idle_timeout:
                loaded = get_loaded_pipelines()
                if loaded:
                    logger.info(f"Idle timeout ({idle_time:.0f}s), unloading {len(loaded)} pipelines")
                    unload_all()
                    self._last_activity = time.time()  # Reset after unload

idle_monitor = IdleMonitor()
