"""LoRA manager for downloading, caching, and applying LoRA weights."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class LoRAState:
    """State tracking for loaded and fused LoRA adapters.
    
    CRITICAL: Fused LoRAs modify base model weights directly.
    We must track fusion state to properly unfuse before any changes.
    """
    # Set of currently loaded adapter names
    loaded_adapters: set[str] = field(default_factory=set)
    
    # Set of adapters currently fused into model weights
    fused_adapters: set[str] = field(default_factory=set)
    
    # Signature of last request to avoid unnecessary unfuse/refuse cycles
    last_request_signature: str = "[]"
    
    # Flag indicating pipeline may have corrupted weights from failed unfusion
    possibly_corrupted: bool = False
    
    # Base weight fingerprint for detecting weight poisoning
    _base_weight_fingerprint: Optional[str] = None
    
    # Safety mode: if True, never fuse LoRAs (slower but safer for untrusted LoRAs)
    safety_mode: bool = False
    
    def clear(self) -> None:
        """Clear all LoRA state."""
        self.loaded_adapters.clear()
        self.fused_adapters.clear()
        self.last_request_signature = "[]"
        self.possibly_corrupted = False
        # Note: _base_weight_fingerprint is NOT cleared - it persists for the checkpoint
    
    def mark_fused(self, adapter_names: set[str]) -> None:
        """Mark adapters as fused into model weights."""
        self.fused_adapters = adapter_names.copy()
    
    def mark_unfused(self) -> None:
        """Mark all adapters as unfused."""
        self.fused_adapters.clear()
    
    def mark_possibly_corrupted(self, reason: str = "") -> None:
        """Mark pipeline as possibly corrupted - requires full unload."""
        self.possibly_corrupted = True
        if reason:
            logger.warning(f"Pipeline marked corrupted: {reason}")
    
    def is_fused(self) -> bool:
        """Check if any adapters are currently fused."""
        return len(self.fused_adapters) > 0
    
    def compute_signature(self, loras: list[dict] | None) -> str:
        """Compute stable signature for LoRA request to detect changes."""
        if not loras:
            return "[]"
        normalized = []
        for item in loras:
            url = item.get("url")
            if not url:
                continue
            strength = round(float(item.get("strength", 1.0)), 4)
            normalized.append({"url": url, "strength": strength})
        normalized.sort(key=lambda x: x["url"])
        return json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)


class LoRAManager:
    """Singleton manager for LoRA weights."""
    
    _instance: Optional["LoRAManager"] = None
    
    def __new__(cls) -> "LoRAManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = get_config()
        self._cache_dir = Path(self._config.lora_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._state = LoRAState()
        logger.info(f"LoRAManager initialized with cache: {self._cache_dir}")
    
    @property
    def state(self) -> LoRAState:
        """Get the LoRA state tracker."""
        return self._state
    
    @property
    def _loaded_adapters(self) -> set[str]:
        """Backward-compatible access to loaded adapters via state."""
        return self._state.loaded_adapters
    
    def get_adapter_name(self, url: str) -> str:
        """Generate adapter name from URL using MD5 hash."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def download_lora(self, url: str, max_retries: int = 3, timeout: int = 120) -> Path:
        """Download LoRA file to cache with atomic write pattern.
        
        Uses temp file + atomic rename to prevent partial/corrupted files.
        Streams download with chunks for memory efficiency.
        
        Args:
            url: URL to download from
            max_retries: Number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Path to cached LoRA file
            
        Raises:
            RuntimeError: If download fails after all retries
        """
        adapter_name = self.get_adapter_name(url)
        cache_path = self._cache_dir / f"{adapter_name}.safetensors"
        temp_path = self._cache_dir / f"{adapter_name}.downloading"
        
        # Return cached file if exists
        if cache_path.exists():
            logger.debug(f"LoRA cached: {adapter_name}")
            return cache_path
        
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading LoRA: {url} (attempt {attempt + 1}/{max_retries})")
                
                # Stream download to temp file
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()
                
                # Write to temp file in chunks (32KB chunks for efficiency)
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                
                # Atomic rename to final path
                temp_path.rename(cache_path)
                logger.info(f"Downloaded LoRA to cache: {adapter_name}")
                return cache_path
                
            except Exception as e:
                last_error = e
                logger.warning(f"Download failed (attempt {attempt + 1}): {e}")
                
                # Clean up partial download
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                
                # Exponential backoff between retries
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed to download LoRA after {max_retries} attempts: {url}") from last_error
    
    def _fuse_adapters(self, pipeline, adapter_names: list[str]) -> bool:
        """Fuse adapters into pipeline for performance.
        
        Args:
            pipeline: The diffusers pipeline
            adapter_names: List of adapter names to fuse
            
        Returns:
            True if fusion succeeded, False if skipped or failed
        """
        if self._state.safety_mode:
            logger.info("LoRA safety mode: skipping fusion (LoRAs applied but not fused)")
            return False
        
        try:
            start = time.monotonic()
            pipeline.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            self._state.mark_fused(set(adapter_names))
            logger.info(f"Fused {len(adapter_names)} LoRAs for performance in {time.monotonic() - start:.2f}s")
            return True
        except Exception as e:
            logger.warning(f"Could not fuse LoRAs (will use unfused): {e}")
            self._state.mark_unfused()
            return False

    def manage_lora_state(self, pipeline, incoming_loras: list[dict] | None) -> bool:
        """Manage LoRA state for a generation request with proper fusion handling.
        
        CRITICAL: This properly handles fusion/unfusion to prevent LoRA weights
        from "poisoning" the base model.
        
        Flow:
        1. Check signature to skip no-op calls (avoids unnecessary unfuse/refuse)
        2. ALWAYS unfuse before making any changes
        3. Unload LoRAs that are no longer needed
        4. Load new LoRAs that are requested
        5. Set active adapters with weights
        6. Fuse for performance
        
        Args:
            pipeline: The diffusers pipeline to manage
            incoming_loras: List of LoRA configs [{"url": "...", "strength": 1.0}, ...]
            
        Returns:
            True if LoRA state changed, False if no changes needed
        """
        import warnings
        import gc
        import torch
        
        # Compute signature to detect if request is identical to last
        sig = self._state.compute_signature(incoming_loras)
        
        # Parse incoming LoRAs
        target_adapters: dict[str, dict] = {}
        target_names: list[str] = []
        target_weights: list[float] = []
        
        if incoming_loras:
            for item in incoming_loras:
                url = item.get("url")
                strength = float(item.get("strength", 1.0))
                if url:
                    adapter_name = self.get_adapter_name(url)
                    target_adapters[adapter_name] = {"url": url, "strength": strength}
                    target_names.append(adapter_name)
                    target_weights.append(strength)
        
        target_names_set = set(target_names)
        
        # Fast path: if signature matches and state is correct, skip all work
        if (
            self._state.last_request_signature == sig and
            target_names_set == self._state.loaded_adapters and
            (not target_names or self._state.is_fused())
        ):
            logger.debug("LoRA request matches cached state, skipping")
            return False
        
        # === STEP 1: UNFUSE BEFORE ANY CHANGES ===
        # This is CRITICAL to prevent weight poisoning
        self.ensure_unfused(pipeline)
        
        # === STEP 2: UNLOAD UNUSED LORAS ===
        to_unload = self._state.loaded_adapters - target_names_set
        if to_unload:
            logger.info(f"Unloading {len(to_unload)} unused LoRAs")
            for name in to_unload:
                try:
                    pipeline.delete_adapters(name)
                    self._state.loaded_adapters.discard(name)
                except Exception as e:
                    logger.warning(f"Failed to delete adapter {name}: {e}")
            
            # Clean up VRAM after unloading
            gc.collect()
            torch.cuda.empty_cache()
        
        # === STEP 3: LOAD NEW LORAS ===
        to_load = target_names_set - self._state.loaded_adapters
        if to_load:
            logger.info(f"Loading {len(to_load)} new LoRAs")
            
            # Check for stale adapters in pipeline (state out of sync)
            existing = self.get_existing_adapters(pipeline)
            
            for name in to_load:
                try:
                    # Remove stale adapter if exists
                    if name in existing:
                        logger.warning(f"Removing stale adapter {name}")
                        try:
                            pipeline.delete_adapters(name)
                        except Exception:
                            pass
                    
                    # Download and load
                    url = target_adapters[name]["url"]
                    start = time.monotonic()
                    path = self.download_lora(url)
                    download_secs = time.monotonic() - start

                    # Suppress harmless LoRA loading warnings
                    load_start = time.monotonic()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="No LoRA keys associated to")
                        warnings.filterwarnings("ignore", message="Already found a `peft_config`")
                        pipeline.load_lora_weights(str(path), adapter_name=name)

                    self._state.loaded_adapters.add(name)
                    logger.info(
                        f"Loaded LoRA: {name} (download {download_secs:.2f}s, "
                        f"load_lora_weights {time.monotonic() - load_start:.2f}s)"
                    )
                    
                except Exception as e:
                    logger.error(f"Error loading LoRA {name}: {e}")
                    # Ensure device consistency after failure
                    self._ensure_pipeline_on_device(pipeline)
                    raise RuntimeError(f"Failed to load LoRA {name}: {e}")
        
        # === STEP 4: SET ACTIVE ADAPTERS ===
        if target_names:
            pipeline.set_adapters(target_names, adapter_weights=target_weights)
            
            # === STEP 5: FUSE FOR PERFORMANCE ===
            self._fuse_adapters(pipeline, target_names)
        
        # Record signature for future fast-path checks
        self._state.last_request_signature = sig
        return True

    # Keep load_loras as an alias for backward compatibility
    def load_loras(self, pipeline, loras: list[dict]) -> None:
        """Load LoRAs - wrapper for manage_lora_state for backward compat."""
        self.manage_lora_state(pipeline, loras)

    def _ensure_pipeline_on_device(self, pipeline) -> None:
        """Ensure all pipeline components are on CUDA after LoRA failure."""
        import torch
        target_device = torch.device("cuda")
        
        components = ['unet', 'vae', 'text_encoder', 'text_encoder_2']
        for comp_name in components:
            comp = getattr(pipeline, comp_name, None)
            if comp is not None:
                try:
                    comp_device = next(comp.parameters()).device
                    if comp_device != target_device:
                        logger.warning(f"Moving {comp_name} from {comp_device} to cuda")
                        comp.to("cuda")
                except StopIteration:
                    pass  # No parameters
                except Exception as e:
                    logger.warning(f"Could not check/move {comp_name}: {e}")
    
    def clear_loras(self, pipeline) -> None:
        """Remove all LoRA adapters from pipeline."""
        if not self._loaded_adapters:
            return
        
        try:
            pipeline.unfuse_lora()
            logger.info("Unfused LoRA weights")
        except Exception as e:
            logger.warning(f"Error unfusing LoRA: {e}")
        
        for adapter_name in list(self._loaded_adapters):
            try:
                pipeline.delete_adapters(adapter_name)
                self._loaded_adapters.discard(adapter_name)
            except Exception as e:
                logger.warning(f"Error deleting adapter {adapter_name}: {e}")
        
        self._loaded_adapters.clear()
        logger.info("Cleared all LoRA adapters")
    
    def reset_loaded_adapters(self) -> None:
        """Reset loaded adapter tracking. Call when pipeline is unloaded."""
        if self._state.loaded_adapters:
            logger.info(f"Clearing LoRA state ({len(self._state.loaded_adapters)} adapters tracked)")
        self._state.clear()
    
    def ensure_unfused(self, pipeline) -> bool:
        """CRITICAL: Ensure all LoRAs are unfused before making changes.
        
        This prevents "poisoned" models where fused LoRA weights stay in base model.
        When LoRAs are fused, their weights are added directly to the model.
        If we don't unfuse before removing/changing LoRAs, those weights stay permanently.
        
        Args:
            pipeline: The diffusers pipeline to unfuse
            
        Returns:
            True if unfusion was performed, False if already unfused
            
        Raises:
            RuntimeError: If unfusion fails and model may be corrupted
        """
        if not self._state.is_fused():
            logger.debug(f"LoRA state: not fused (loaded={len(self._state.loaded_adapters)})")
            return False
        
        expected = set(self._state.fused_adapters)
        logger.info(f"Unfusing {len(expected)} adapters: {sorted(expected)}")
        
        try:
            pipeline.unfuse_lora()
            logger.info(f"Unfused {len(expected)} LoRAs from model weights")
            self._state.mark_unfused()
            
            # Verify unfusion succeeded
            if not self._verify_unfusion_clean(pipeline):
                self._state.mark_possibly_corrupted("Post-unfuse verification failed")
                raise RuntimeError(f"LoRA unfuse completed but verification failed")
            
            logger.debug("Unfusion verified clean")
            return True
            
        except RuntimeError:
            raise
        except Exception as e:
            msg = str(e).lower()
            # Check for benign "not fused" errors
            benign = any(k in msg for k in ["not fused", "no lora", "nothing to unfuse", "no adapters"])
            
            # Clear state even on benign error (tracking unreliable)
            self._state.mark_unfused()
            
            if benign:
                logger.warning(f"Unfuse reported already-unfused (benign): {e}")
                return False
            
            # Non-benign failure: mark corrupted
            self._state.mark_possibly_corrupted(f"Unfuse exception: {e}")
            raise RuntimeError(f"LoRA unfuse failed: {e}")

    def _verify_unfusion_clean(self, pipeline) -> bool:
        """Verify that no fused adapters remain in the pipeline.
        
        Checks internal state of pipeline components to ensure unfusion
        actually removed the fused weights. Also verifies weight fingerprint
        matches baseline if available.
        
        Returns:
            True if pipeline appears clean, False if corruption detected
        """
        try:
            # Check for VAE in exclusion - VAE should NEVER have LoRA applied
            vae = getattr(pipeline, 'vae', None)
            if vae is not None and hasattr(vae, 'peft_config') and vae.peft_config:
                logger.error("VAE has LoRA adapters - this should never happen!")
                return False
            
            for component_name in ['unet', 'text_encoder', 'text_encoder_2']:
                component = getattr(pipeline, component_name, None)
                if component is None:
                    continue
                
                # Check for _lora_scale attribute indicating fused state
                if hasattr(component, '_lora_scale') and component._lora_scale != 1.0:
                    logger.error(f"{component_name} has non-default _lora_scale: {component._lora_scale}")
                    return False
                
                # Check peft internal state for fused indicators
                if hasattr(component, 'peft_config'):
                    for adapter_name, config in component.peft_config.items():
                        if hasattr(config, 'merge_weights') and config.merge_weights:
                            logger.error(f"{component_name}.{adapter_name} still has merge_weights=True")
                            return False
            
            # Verify weight fingerprint matches baseline (if baseline exists)
            if not self.verify_weights_clean(pipeline):
                logger.error("Weight fingerprint mismatch - base weights may be poisoned!")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Unfusion verification error: {e}")
            # On verification error, assume clean to avoid false positives
            return True
    
    def _compute_weight_fingerprint(self, pipeline) -> str:
        """Compute lightweight fingerprint of base model weights.
        
        Uses first/last few elements of key tensors to detect changes
        without hashing entire model (too slow).
        
        Args:
            pipeline: The diffusers pipeline
            
        Returns:
            String fingerprint of sampled weight values
        """
        import torch
        
        fingerprint_parts = []
        for name in ['unet', 'text_encoder']:
            component = getattr(pipeline, name, None)
            if component is None:
                continue
            # Get first conv/linear layer's weight hash
            for module in component.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.data
                    # Use first 10 and last 10 values as fingerprint
                    flat = w.flatten()
                    sample_size = min(10, flat.numel())
                    sample = torch.cat([flat[:sample_size], flat[-sample_size:]])
                    fingerprint_parts.append(sample.sum().item())
                    break
        return str(fingerprint_parts)
    
    def store_base_fingerprint(self, pipeline) -> None:
        """Store the base weight fingerprint for later verification.
        
        Call this after loading a checkpoint, BEFORE applying any LoRAs.
        
        Args:
            pipeline: The freshly loaded pipeline
        """
        fingerprint = self._compute_weight_fingerprint(pipeline)
        self._state._base_weight_fingerprint = fingerprint
        logger.info(f"Stored base weight fingerprint: {fingerprint[:50]}...")
    
    def verify_weights_clean(self, pipeline) -> bool:
        """Verify weights match the original baseline fingerprint.
        
        Args:
            pipeline: The pipeline to check
            
        Returns:
            True if weights match baseline (or no baseline exists), False if mismatch
        """
        if self._state._base_weight_fingerprint is None:
            return True  # No baseline to compare
        current = self._compute_weight_fingerprint(pipeline)
        if current != self._state._base_weight_fingerprint:
            logger.error(f"Weight fingerprint mismatch!")
            logger.error(f"  Expected: {self._state._base_weight_fingerprint}")
            logger.error(f"  Current:  {current}")
            return False
        return True
    
    def set_safety_mode(self, enabled: bool) -> None:
        """Enable or disable safety mode.
        
        When safety mode is enabled, LoRAs are never fused into the model.
        This is slower but safer for untrusted LoRAs since unfusion bugs
        cannot cause weight poisoning.
        
        Args:
            enabled: True to enable safety mode, False to disable
        """
        self._state.safety_mode = enabled
        logger.info(f"LoRA safety mode {'enabled' if enabled else 'disabled'}")

    def get_existing_adapters(self, pipeline) -> set[str]:
        """Get all adapter names currently loaded in the pipeline.
        
        Args:
            pipeline: The diffusers pipeline to check
            
        Returns:
            Set of adapter names
        """
        adapters = set()
        for component_name in ['unet', 'text_encoder', 'text_encoder_2']:
            component = getattr(pipeline, component_name, None)
            if component is not None and hasattr(component, 'peft_config'):
                adapters.update(component.peft_config.keys())
        return adapters

    def clear_all_adapters(self, pipeline) -> None:
        """CRITICAL: Completely clear all adapters from the pipeline.
        
        Ensures clean slate by:
        1. Unfusing any fused LoRAs (MUST do this first!)
        2. Deleting all loaded adapters
        3. Clearing state tracking
        
        Args:
            pipeline: The diffusers pipeline to clear
            
        Raises:
            RuntimeError: If unfusion fails or critical cleanup fails
        """
        import gc
        import torch
        
        logger.info(f"Clearing adapters (loaded={len(self._state.loaded_adapters)}, "
                    f"fused={len(self._state.fused_adapters)})")
        
        # Step 1: UNFUSE FIRST (critical!)
        self.ensure_unfused(pipeline)
        
        # Step 2: Get all adapters from pipeline (may include orphaned ones)
        all_adapters = self.get_existing_adapters(pipeline)
        
        # Step 3: Delete all adapters
        failed_deletes = []
        if all_adapters:
            logger.info(f"Deleting {len(all_adapters)} adapters from pipeline")
            for adapter_name in all_adapters:
                try:
                    pipeline.delete_adapters(adapter_name)
                except Exception as e:
                    logger.warning(f"Could not delete adapter {adapter_name}: {e}")
                    failed_deletes.append(adapter_name)
        
        # Step 4: Clear state
        self._state.clear()
        
        # Step 5: Clean up VRAM
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 6: If any deletes failed, mark as possibly corrupted
        if failed_deletes:
            self._state.mark_possibly_corrupted(f"Failed to delete adapters: {failed_deletes}")
            raise RuntimeError(f"Failed to delete adapters: {sorted(failed_deletes)}")
        
        logger.info("All adapters cleared")

    def prepare_for_checkpoint_switch(self, pipeline) -> bool:
        """Prepare the pipeline for a checkpoint switch.
        
        CRITICAL: Must be called before loading a new checkpoint to ensure
        no LoRA weights are left in the model.
        
        Args:
            pipeline: The current pipeline
            
        Returns:
            True if pipeline is clean and safe for checkpoint switch
            
        Raises:
            RuntimeError: If cleanup fails or corruption detected
        """
        logger.info("Preparing for checkpoint switch...")
        
        # Check if already marked corrupted
        if self._state.possibly_corrupted:
            logger.error("Pipeline already marked as possibly corrupted - forcing full unload")
            raise RuntimeError("Pipeline marked corrupted from previous unfusion failure")
        
        logger.info(f"LoRA state: loaded={len(self._state.loaded_adapters)}, "
                    f"fused={len(self._state.fused_adapters)}, "
                    f"corrupted={self._state.possibly_corrupted}")
        
        try:
            self.clear_all_adapters(pipeline)
        except RuntimeError:
            raise
        except Exception as e:
            self._state.mark_possibly_corrupted(f"Exception during adapter clear: {e}")
            raise RuntimeError(f"Failed to clear adapters for checkpoint switch: {e}")
        
        # Final check - verify no adapters remain
        remaining = self.get_existing_adapters(pipeline)
        if remaining:
            self._state.mark_possibly_corrupted(f"Adapters remain after clear: {remaining}")
            raise RuntimeError(f"Adapters still present after clear: {sorted(remaining)}")
        
        logger.info("Pipeline prepared for checkpoint switch (clean)")
        return True

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
