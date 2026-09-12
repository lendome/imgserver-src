"""Pipeline registry with lazy loading, LRU tracking, and optimized checkpoint switching."""

import gc
import logging
import threading
import time
from typing import Dict, Optional, Type

import torch

from .base import BasePipeline, get_module_info
from .module_loader import ensure_modules_loaded
from .sdxl import SDXLPipeline
from .zimg import ZImagePipeline
from .tts import TTSPipeline
from .ltx import LTXPipeline
from .wan import WanPipeline
from ..vram.manager import vram_manager
from ..vram.ram_manager import ram_manager

logger = logging.getLogger(__name__)

# Thread safety lock for pipeline operations
_registry_lock = threading.RLock()  # RLock allows reentrant calls

# Pipeline type registry
# Will be populated dynamically from module metadata, with fallback hardcoded entries
_PIPELINE_TYPES: Dict[str, Type[BasePipeline]] = {
    "sdxl": SDXLPipeline,
    "zimg": ZImagePipeline,
    "tts": TTSPipeline,
    "ltx": LTXPipeline,
    "wan": WanPipeline,
}

# Loaded pipeline instances (on GPU)
# Key format: "type" for zimg, "type:checkpoint" for sdxl
_loaded_pipelines: Dict[str, BasePipeline] = {}

# Parked pipeline instances (on CPU)
_parked_pipelines: Dict[str, BasePipeline] = {}

# Current active checkpoint for each pipeline type
_active_checkpoints: Dict[str, Optional[str]] = {}

# Flag to track if _PIPELINE_TYPES has been populated dynamically
_PIPELINE_TYPES_INITIALIZED = False


def _initialize_pipeline_types() -> None:
    """
    Dynamically populate _PIPELINE_TYPES from module metadata.
    
    This function:
    1. Calls ensure_modules_loaded() to ensure all modules are discovered
    2. Iterates through registered modules to find those with pipeline classes
    3. Updates _PIPELINE_TYPES with dynamic entries
    4. Falls back to existing hardcoded entries if module info not found
    
    This is called once during first pipeline access to avoid circular imports.
    """
    global _PIPELINE_TYPES_INITIALIZED
    
    if _PIPELINE_TYPES_INITIALIZED:
        return
    
    logger.debug("Initializing pipeline types from module metadata...")
    try:
        ensure_modules_loaded()
        _PIPELINE_TYPES_INITIALIZED = True
        logger.debug(f"Pipeline types initialized. Current registry: {list(_PIPELINE_TYPES.keys())}")
    except Exception as e:
        logger.warning(f"Failed to initialize pipeline types from modules: {e}. Using hardcoded fallback.")
        _PIPELINE_TYPES_INITIALIZED = True


def _unload_conflicting_pipelines(pipeline_name: str) -> None:
    """
    Unload pipelines that conflict with the requested pipeline type.
    
    This function dynamically determines conflicts based on module metadata rather than
    hardcoded if/elif logic. For each conflicting pipeline type:
    - Unloads all loaded pipelines that start with that conflict name
    - Unloads all parked pipelines that start with that conflict name
    - Performs gc.collect() and torch.cuda.empty_cache() if anything was unloaded
    
    Falls back to hardcoded behavior if module info is not found (backward compatibility).
    
    Args:
        pipeline_name: The pipeline type being loaded (e.g., "zimg", "sdxl", "tts")
    """
    module_info = get_module_info(pipeline_name)
    
    if not module_info:
        # Fallback to hardcoded logic for backward compatibility
        logger.debug(f"No module info found for '{pipeline_name}', using fallback conflict resolution")
        return
    
    # Get list of conflicting pipeline types
    conflicts = module_info.conflicts_with
    if not conflicts:
        logger.debug(f"No conflicts defined for '{pipeline_name}'")
        return
    
    anything_unloaded = False
    
    for conflict_name in conflicts:
        # Unload all loaded pipelines starting with this conflict name
        loaded_to_remove = [k for k in list(_loaded_pipelines.keys()) if k.startswith(conflict_name)]
        for key in loaded_to_remove:
            logger.info(f"Unloading {conflict_name} pipeline '{key}' before loading {pipeline_name}")
            old_pipeline = _loaded_pipelines.pop(key)
            old_pipeline.unload()
            anything_unloaded = True
        
        # Unload all parked pipelines starting with this conflict name
        parked_to_remove = [k for k in list(_parked_pipelines.keys()) if k.startswith(conflict_name)]
        for key in parked_to_remove:
            logger.info(f"Unloading parked {conflict_name} pipeline '{key}'")
            parked = _parked_pipelines.pop(key)
            parked.unload()
            anything_unloaded = True
    
    # Clean up VRAM if anything was unloaded
    if anything_unloaded:
        gc.collect()
        torch.cuda.empty_cache()



def _get_pipeline_key(name: str, checkpoint: Optional[str] = None) -> str:
    """Get registry key for a pipeline."""
    if name == "sdxl" and checkpoint:
        return f"sdxl:{checkpoint}"
    if name == "zimg" and checkpoint:
        return f"zimg:{checkpoint}"
    return name


def get_pipeline(name: str, checkpoint: Optional[str] = None, **kwargs) -> BasePipeline:
    """
    Get a pipeline by name, loading it if necessary.
    
    For SDXL: Handles checkpoint switching efficiently.
    For ZImg: Ignores checkpoint (uses fixed model).
    
    Args:
        name: Pipeline type ("sdxl" or "zimg")
        checkpoint: Checkpoint filename (for SDXL)
        **kwargs: Additional arguments passed to pipeline constructor
        
    Returns:
        Loaded pipeline instance
    """
    with _registry_lock:
        # Ensure pipeline types are initialized
        _initialize_pipeline_types()
        
        if name not in _PIPELINE_TYPES:
            raise ValueError(f"Unknown pipeline: {name}. Available: {list(_PIPELINE_TYPES.keys())}")
        
        # For TTS, checkpoint is ignored (TTS doesn't use checkpoints)
        if name == "tts":
            checkpoint = None
        
        key = _get_pipeline_key(name, checkpoint)
        
        # Check if exact pipeline is already loaded
        if key in _loaded_pipelines:
            pipeline = _loaded_pipelines[key]
            pipeline.touch()  # Update LRU
            return pipeline
        
        # CRITICAL: When loading a different pipeline type (e.g., zimg after sdxl),
        # we need to unload the other type first to prevent VRAM overflow into shared memory.
        # This is especially important because SDXL (~7GB) + ZImg (~3GB) = ~10GB which can
        # exceed available VRAM on some cards when combined with other allocations.
        # This MUST happen BEFORE any checkpoint switching logic.
        # Use metadata-driven conflict resolution instead of hardcoded if/elif blocks.
        _unload_conflicting_pipelines(name)

        
        # For SDXL: Check if we have a different checkpoint loaded
        if name == "sdxl":
            # Find any loaded SDXL pipeline
            sdxl_keys = [k for k in _loaded_pipelines.keys() if k.startswith("sdxl")]
            if sdxl_keys:
                old_key = sdxl_keys[0]
                old_pipeline = _loaded_pipelines[old_key]
                old_checkpoint = getattr(old_pipeline, '_checkpoint', None)

                # Fast path: swap weights in place, preserving torch.compile
                # (no dynamo retrace / inductor recompile on first generation).
                if hasattr(old_pipeline, "switch_checkpoint"):
                    try:
                        swap_start = time.time()
                        old_pipeline.switch_checkpoint(checkpoint)
                        _loaded_pipelines.pop(old_key, None)
                        _loaded_pipelines[key] = old_pipeline
                        old_pipeline.touch()
                        _active_checkpoints["sdxl"] = checkpoint
                        logger.info(
                            f"SDXL checkpoint hot-swapped {old_checkpoint} -> {checkpoint} "
                            f"in {time.time() - swap_start:.2f}s (compile preserved)"
                        )
                        return old_pipeline
                    except Exception as e:
                        # Weights may be partially swapped - pipeline is unusable.
                        # Fall back to the full unload + reload path below.
                        logger.warning(f"Fast checkpoint swap failed ({e}); falling back to full reload")
                        try:
                            old_pipeline.unload()
                        except Exception as unload_err:
                            logger.warning(f"Unload after failed swap also failed: {unload_err}")
                        _loaded_pipelines.pop(old_key, None)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                # Need to switch checkpoint
                logger.info(f"Switching SDXL checkpoint: {old_checkpoint} -> {checkpoint}")
                start_time = time.time()
                
                # Fast unload: component-by-component cleanup frees VRAM faster
                # This avoids the slow full .to("cpu") move
                old_pipeline.unload()
                # pop, not del: the failed-swap fallback above may have already
                # removed this key (double-unload bookkeeping must be idempotent)
                _loaded_pipelines.pop(old_key, None)
                
                # Force VRAM cleanup between unload and load
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                unload_time = time.time() - start_time
                logger.info(f"Old checkpoint unloaded in {unload_time:.2f}s")
                
                # Load new checkpoint
                load_start = time.time()
                pipeline = SDXLPipeline(checkpoint=checkpoint)
                pipeline.load()
                _loaded_pipelines[key] = pipeline
                _active_checkpoints["sdxl"] = checkpoint
                
                load_time = time.time() - load_start
                total_time = time.time() - start_time
                logger.info(f"SDXL checkpoint switched in {total_time:.2f}s (unload: {unload_time:.2f}s, load: {load_time:.2f}s)")
                return pipeline
            
            # Check parked SDXL pipelines
            sdxl_parked = [k for k in _parked_pipelines.keys() if k.startswith("sdxl")]
            for parked_key in sdxl_parked:
                parked_pipeline = _parked_pipelines[parked_key]
                parked_checkpoint = getattr(parked_pipeline, '_checkpoint', None)
                
                if parked_checkpoint == checkpoint:
                    # Restore matching parked pipeline
                    _parked_pipelines.pop(parked_key)
                    needed = parked_pipeline.estimate_vram()
                    
                    if not vram_manager.ensure_free(needed, _park_or_unload_lru):
                        raise RuntimeError(f"Cannot restore {name}: not enough VRAM")
                    
                    parked_pipeline.to_gpu()
                    _loaded_pipelines[key] = parked_pipeline
                    logger.info(f"SDXL pipeline restored from CPU: {checkpoint}")
                    return parked_pipeline
                else:
                    # Different checkpoint parked, unload it
                    parked_pipeline.unload()
                    _parked_pipelines.pop(parked_key, None)
        
        # Check if pipeline is parked on CPU (for non-checkpoint cases)
        if key in _parked_pipelines:
            pipeline = _parked_pipelines.pop(key)
            needed = pipeline.estimate_vram()
            
            if not vram_manager.ensure_free(needed, _park_or_unload_lru):
                raise RuntimeError(f"Cannot restore {name}: not enough VRAM")
            
            pipeline.to_gpu()
            _loaded_pipelines[key] = pipeline
            logger.info(f"Pipeline '{key}' restored from CPU")
            return pipeline
        
        # Create new instance
        pipeline_class = _PIPELINE_TYPES[name]
        if name == "sdxl":
            pipeline = pipeline_class(checkpoint=checkpoint, **kwargs)
        elif name == "zimg" and checkpoint:
            # Z-Image can load from local checkpoint files
            pipeline = pipeline_class(checkpoint=checkpoint, **kwargs)
        else:
            pipeline = pipeline_class(**kwargs)
        
        # Ensure we have VRAM
        needed = pipeline.estimate_vram()
        if not vram_manager.ensure_free(needed, _park_or_unload_lru):
            raise RuntimeError(f"Cannot load {name}: not enough VRAM")
        
        # Load and register
        pipeline.load()
        _loaded_pipelines[key] = pipeline
        if name == "sdxl":
            _active_checkpoints["sdxl"] = checkpoint
        logger.info(f"Pipeline '{key}' loaded and registered")
        
        return pipeline


def park_pipeline(key: str) -> bool:
    """
    Park a pipeline to CPU RAM instead of fully unloading.
    
    Args:
        key: Pipeline key (e.g., "zimg" or "sdxl:checkpoint.safetensors")
        
    Returns:
        True if pipeline was parked, False if not loaded or parking failed
    """
    with _registry_lock:
        if key not in _loaded_pipelines:
            return False
        
        pipeline = _loaded_pipelines[key]
        model_size = pipeline.estimate_vram()
        
        # Check if we have enough RAM to park
        if not ram_manager.can_park_model(model_size):
            logger.warning(f"Not enough RAM to park '{key}', will unload instead")
            return False
        
        pipeline.to_cpu()
        _loaded_pipelines.pop(key)
        _parked_pipelines[key] = pipeline
        logger.info(f"Pipeline '{key}' parked to CPU")
        return True


def unload_pipeline(key: str) -> bool:
    """
    Unload a specific pipeline (from GPU or CPU).
    
    Args:
        key: Pipeline key to unload
        
    Returns:
        True if pipeline was unloaded, False if not loaded
    """
    with _registry_lock:
        # Check loaded pipelines first
        if key in _loaded_pipelines:
            pipeline = _loaded_pipelines.pop(key)
            pipeline.unload()
            logger.info(f"Pipeline '{key}' unloaded from GPU")
            return True
        
        # Check parked pipelines
        if key in _parked_pipelines:
            pipeline = _parked_pipelines.pop(key)
            pipeline.unload()
            logger.info(f"Pipeline '{key}' unloaded from CPU")
            return True
        
        return False


def unload_all() -> int:
    """
    Unload all pipelines (both loaded and parked).
    
    Returns:
        Number of pipelines unloaded
    """
    with _registry_lock:
        count = 0
        for name in list(_loaded_pipelines.keys()):
            if unload_pipeline(name):
                count += 1
        for name in list(_parked_pipelines.keys()):
            if unload_pipeline(name):
                count += 1
        return count


def get_loaded_pipelines() -> Dict[str, BasePipeline]:
    """Get dict of currently loaded pipelines (on GPU)."""
    return dict(_loaded_pipelines)


def get_parked_pipelines() -> Dict[str, BasePipeline]:
    """Get dict of currently parked pipelines (on CPU)."""
    return dict(_parked_pipelines)


def _get_lru_pipeline() -> Optional[str]:
    """Get name of least-recently-used loaded pipeline."""
    if not _loaded_pipelines:
        return None
    
    return min(_loaded_pipelines.keys(), key=lambda n: _loaded_pipelines[n].last_used)


def _park_or_unload_lru() -> bool:
    """
    Park the LRU pipeline to CPU if RAM available, otherwise unload it.
    
    Returns:
        True if a pipeline was parked/unloaded, False if nothing to evict
    """
    lru = _get_lru_pipeline()
    if lru is None:
        return False
    
    # Try to park first
    if park_pipeline(lru):
        logger.info(f"Auto-parked LRU pipeline: {lru}")
        return True
    
    # Fall back to unload
    logger.info(f"Auto-unloading LRU pipeline: {lru}")
    return unload_pipeline(lru)


def _unload_lru() -> bool:
    """
    Unload the least-recently-used pipeline.
    
    Returns:
        True if a pipeline was unloaded, False if nothing to unload
    """
    lru = _get_lru_pipeline()
    if lru is None:
        return False
    
    logger.info(f"Auto-unloading LRU pipeline: {lru}")
    return unload_pipeline(lru)


def get_active_checkpoint(pipeline_type: str) -> Optional[str]:
    """Get the currently active checkpoint for a pipeline type."""
    return _active_checkpoints.get(pipeline_type)
