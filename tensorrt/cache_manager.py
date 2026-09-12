"""TensorRT engine cache manager with hash-based invalidation."""
import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class EngineMetadata:
    """Metadata for a cached TensorRT engine."""
    model_hash: str
    config_hash: str
    cuda_version: str
    tensorrt_version: str
    gpu_name: str
    compute_capability: Tuple[int, int]
    precision: str
    created_at: str
    input_shapes: Dict[str, list]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'EngineMetadata':
        d['compute_capability'] = tuple(d['compute_capability'])
        return cls(**d)

class TensorRTEngineCache:
    """Thread-safe TensorRT engine cache with automatic invalidation.
    
    Engines are cached based on a composite hash of:
    - Model weights hash
    - TensorRT config (precision, workspace, etc.)
    - CUDA version
    - TensorRT version
    - GPU compute capability
    """
    
    _instance: Optional['TensorRTEngineCache'] = None
    _lock = threading.Lock()
    
    def __init__(self, cache_dir: str = "tensorrt_engines"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timing_cache_path = self._cache_dir / "timing.cache"
        self._file_lock = threading.Lock()
        
        # In-memory cache of loaded engines
        self._loaded_engines: Dict[str, Any] = {}
        
        logger.info(f"TensorRT engine cache initialized at {self._cache_dir}")
    
    @classmethod
    def get_instance(cls, cache_dir: str = "tensorrt_engines") -> 'TensorRTEngineCache':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cache_dir)
            return cls._instance
    
    def _compute_model_hash(self, model_config: dict, weights_path: Optional[str] = None) -> str:
        """Compute hash of model configuration and optionally weights file."""
        components = [json.dumps(model_config, sort_keys=True)]
        
        if weights_path and os.path.exists(weights_path):
            # Use file modification time + size as proxy for content
            stat = os.stat(weights_path)
            components.append(f"{weights_path}:{stat.st_size}:{stat.st_mtime}")
        
        return hashlib.sha256("".join(components).encode()).hexdigest()[:16]
    
    def _compute_config_hash(self, tensorrt_config: dict) -> str:
        """Compute hash of TensorRT configuration."""
        config_str = json.dumps(tensorrt_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _get_system_info(self) -> dict:
        """Get current system info for cache validation."""
        import torch
        
        cuda_version = torch.version.cuda or "unknown"
        
        try:
            import tensorrt as trt
            trt_version = trt.__version__
        except ImportError:
            trt_version = "unknown"
        
        gpu_name = "unknown"
        compute_capability = (0, 0)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
        
        return {
            "cuda_version": cuda_version,
            "tensorrt_version": trt_version,
            "gpu_name": gpu_name,
            "compute_capability": compute_capability,
        }
    
    def compute_cache_key(
        self,
        model_name: str,
        model_config: dict,
        tensorrt_config: dict,
        weights_path: Optional[str] = None,
    ) -> str:
        """Compute unique cache key for an engine."""
        model_hash = self._compute_model_hash(model_config, weights_path)
        config_hash = self._compute_config_hash(tensorrt_config)
        sys_info = self._get_system_info()
        
        key_components = [
            model_name,
            model_hash,
            config_hash,
            sys_info["cuda_version"],
            sys_info["tensorrt_version"],
            f"sm{sys_info['compute_capability'][0]}{sys_info['compute_capability'][1]}",
        ]
        
        return "_".join(key_components)
    
    def _get_engine_path(self, cache_key: str) -> Path:
        """Get path for cached engine file."""
        return self._cache_dir / f"{cache_key}.engine"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get path for engine metadata file."""
        return self._cache_dir / f"{cache_key}.json"
    
    def has_cached_engine(self, cache_key: str) -> bool:
        """Check if a valid cached engine exists."""
        engine_path = self._get_engine_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        if not engine_path.exists() or not metadata_path.exists():
            return False
        
        # Validate metadata matches current system
        try:
            with open(metadata_path, 'r') as f:
                metadata = EngineMetadata.from_dict(json.load(f))
            
            sys_info = self._get_system_info()
            
            if metadata.cuda_version != sys_info["cuda_version"]:
                logger.info(f"Cache miss: CUDA version changed ({metadata.cuda_version} -> {sys_info['cuda_version']})")
                return False
            
            if metadata.tensorrt_version != sys_info["tensorrt_version"]:
                logger.info(f"Cache miss: TensorRT version changed ({metadata.tensorrt_version} -> {sys_info['tensorrt_version']})")
                return False
            
            if metadata.compute_capability != sys_info["compute_capability"]:
                logger.info(f"Cache miss: GPU changed ({metadata.gpu_name} -> {sys_info['gpu_name']})")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate cache metadata: {e}")
            return False
    
    def load_engine(self, cache_key: str) -> Optional[bytes]:
        """Load cached engine bytes if valid."""
        with self._file_lock:
            if cache_key in self._loaded_engines:
                logger.debug(f"Returning in-memory cached engine: {cache_key}")
                return self._loaded_engines[cache_key]
            
            if not self.has_cached_engine(cache_key):
                return None
            
            engine_path = self._get_engine_path(cache_key)
            
            try:
                with open(engine_path, 'rb') as f:
                    engine_bytes = f.read()
                
                self._loaded_engines[cache_key] = engine_bytes
                logger.info(f"Loaded cached TensorRT engine: {cache_key}")
                return engine_bytes
                
            except Exception as e:
                logger.error(f"Failed to load cached engine: {e}")
                return None
    
    def save_engine(
        self,
        cache_key: str,
        engine_bytes: bytes,
        model_config: dict,
        tensorrt_config: dict,
        input_shapes: Dict[str, list],
    ) -> bool:
        """Save engine to cache with metadata."""
        with self._file_lock:
            engine_path = self._get_engine_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            try:
                # Save engine
                with open(engine_path, 'wb') as f:
                    f.write(engine_bytes)
                
                # Create and save metadata
                sys_info = self._get_system_info()
                metadata = EngineMetadata(
                    model_hash=self._compute_model_hash(model_config),
                    config_hash=self._compute_config_hash(tensorrt_config),
                    cuda_version=sys_info["cuda_version"],
                    tensorrt_version=sys_info["tensorrt_version"],
                    gpu_name=sys_info["gpu_name"],
                    compute_capability=sys_info["compute_capability"],
                    precision=tensorrt_config.get("precision", "fp16"),
                    created_at=datetime.now().isoformat(),
                    input_shapes=input_shapes,
                )
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                # Update in-memory cache
                self._loaded_engines[cache_key] = engine_bytes
                
                logger.info(f"Saved TensorRT engine to cache: {cache_key}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save engine to cache: {e}")
                # Cleanup partial writes
                engine_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                return False
    
    def invalidate(self, cache_key: str) -> None:
        """Invalidate a specific cached engine."""
        with self._file_lock:
            self._loaded_engines.pop(cache_key, None)
            self._get_engine_path(cache_key).unlink(missing_ok=True)
            self._get_metadata_path(cache_key).unlink(missing_ok=True)
            logger.info(f"Invalidated cached engine: {cache_key}")
    
    def clear_all(self) -> None:
        """Clear entire cache."""
        with self._file_lock:
            self._loaded_engines.clear()
            
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Cleared all cached TensorRT engines")
    
    def get_timing_cache_path(self) -> Path:
        """Get path for TensorRT timing cache."""
        return self._timing_cache_path
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        engine_files = list(self._cache_dir.glob("*.engine"))
        total_size = sum(f.stat().st_size for f in engine_files)
        
        return {
            "cache_dir": str(self._cache_dir),
            "num_engines": len(engine_files),
            "total_size_mb": total_size / (1024 * 1024),
            "in_memory_count": len(self._loaded_engines),
            "timing_cache_exists": self._timing_cache_path.exists(),
        }


def get_engine_cache(cache_dir: str = "tensorrt_engines") -> TensorRTEngineCache:
    """Get the global engine cache instance."""
    return TensorRTEngineCache.get_instance(cache_dir)
