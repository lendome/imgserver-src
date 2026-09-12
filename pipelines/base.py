"""Base pipeline abstract class for all image generation pipelines."""

from abc import ABC, abstractmethod
import time
from typing import Union, Dict, List, Optional
from dataclasses import dataclass, field
from PIL import Image


class BasePipeline(ABC):
    """Abstract base class for image generation pipelines."""
    
    last_used: float = 0.0
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if model is loaded on GPU."""
        pass
    
    @property
    def is_parked(self) -> bool:
        """True if model is parked on CPU (offloaded from GPU)."""
        return False
    
    @abstractmethod
    def load(self) -> None:
        """Load model to GPU."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from GPU, clear references."""
        pass
    
    def to_cpu(self) -> None:
        """Move model to CPU RAM (parking). Default: no-op, subclasses override."""
        pass
    
    def to_gpu(self) -> None:
        """Move model from CPU back to GPU. Default: no-op, subclasses override."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Union[Image.Image, list[Image.Image]]:
        """Run generation. Implementations must call touch()."""
        pass
    
    @abstractmethod
    def estimate_vram(self) -> int:
        """Estimated VRAM needed in bytes."""
        pass
    
    def touch(self) -> None:
        """Update last_used to current time for LRU tracking."""
        self.last_used = time.time()


# ============================================================================
# Module Registration System
# ============================================================================

@dataclass
class ModuleInfo:
    """Metadata for a registered pipeline module."""
    name: str
    display_name: str
    output_type: str  # "image", "video", "audio"
    conflicts_with: List[str] = field(default_factory=list)
    checkpoint_patterns: List[str] = field(default_factory=list)
    vram_estimate_gb: float = 0.0
    supports_checkpoints: bool = True


# Global registry of modules
_MODULE_REGISTRY: Dict[str, ModuleInfo] = {}


def pipeline_module(
    name: str,
    display_name: str,
    output_type: str,
    conflicts_with: Optional[List[str]] = None,
    checkpoint_patterns: Optional[List[str]] = None,
    vram_estimate_gb: float = 0.0,
    supports_checkpoints: bool = True,
):
    """
    Decorator to register a pipeline class as a module.
    
    Args:
        name: Module identifier (e.g., "sdxl", "zimg", "tts", "ltx")
        display_name: Human-readable name (e.g., "SDXL Image Generation")
        output_type: Type of output produced ("image", "video", "audio")
        conflicts_with: List of pipeline names that conflict with this one (VRAM constraints)
        checkpoint_patterns: Regex patterns to detect this model type
        vram_estimate_gb: Estimated VRAM requirement in GB
        supports_checkpoints: Whether this pipeline supports loading different checkpoints
    """
    def decorator(cls):
        module_info = ModuleInfo(
            name=name,
            display_name=display_name,
            output_type=output_type,
            conflicts_with=conflicts_with or [],
            checkpoint_patterns=checkpoint_patterns or [],
            vram_estimate_gb=vram_estimate_gb,
            supports_checkpoints=supports_checkpoints,
        )
        _MODULE_REGISTRY[name] = module_info
        cls._module_info = module_info
        return cls
    
    return decorator


def get_module_info(name: str) -> Optional[ModuleInfo]:
    """Get module info by name."""
    return _MODULE_REGISTRY.get(name)


def get_all_modules() -> Dict[str, ModuleInfo]:
    """Get all registered modules."""
    return _MODULE_REGISTRY.copy()


def get_modules_by_output_type(output_type: str) -> List[ModuleInfo]:
    """Get all modules that produce a specific output type."""
    return [info for info in _MODULE_REGISTRY.values() if info.output_type == output_type]
