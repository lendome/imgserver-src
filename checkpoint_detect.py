"""Checkpoint model type detection based on filename patterns and model structure."""

import os
import re
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Detected checkpoint configuration."""
    model_type: str  # "sdxl", "zimg", "anima"
    prediction_type: str  # "epsilon", "v_prediction"
    is_zimage: bool
    is_anima: bool
    config_source: str  # Where the config came from


# Known model patterns for auto-detection
# Format: (pattern, prediction_type)
# Order matters - more specific patterns should come first!
KNOWN_MODEL_PATTERNS = [
    # v_prediction models (more specific patterns first)
    (r'noob.*xl', 'v_prediction'),
    (r'vpred', 'v_prediction'),
    (r'v-pred', 'v_prediction'),
    (r'v_prediction', 'v_prediction'),
    (r'pony.*v6', 'v_prediction'),
    (r'pdxl', 'v_prediction'),
    (r'illustrious.*xl.*v0\.[1-9]', 'v_prediction'),
    
    # epsilon models (explicit)
    (r'wai.*illustrious', 'epsilon'),
    (r'illustrious', 'epsilon'),
    (r'animagine', 'epsilon'),
    (r'kohaku', 'epsilon'),
    (r'sdxl.*base', 'epsilon'),
    (r'juggernaut', 'epsilon'),
    (r'realvis', 'epsilon'),
    (r'dreamshaper.*xl', 'epsilon'),
    (r'amanatsu', 'epsilon'),
    (r'waiNSFW', 'epsilon'),
    (r'wai.*nsfw', 'epsilon'),
]

# Z-Image model patterns (uses different pipeline)
ZIMAGE_MODEL_PATTERNS = [
    r'zimage',
    r'z-image', 
    r'z_image',
    r'zimg',
    r'z-turbo',
    r'zturbo',
    r'moodyv',
    r'moody.*v\d+',
    r'realdream',  # realDream models are Z-Image based
]

# Anima model patterns (uses Cosmos + Qwen - NOT supported yet)
ANIMA_MODEL_PATTERNS = [
    r'anima',
    r'anima-preview',
    r'circlestone',
    r'cosmos-predict',
    r'cosmos_predict',
]


def detect_model_type_from_keys(checkpoint_path: str) -> Optional[str]:
    """
    Detect model type by inspecting the checkpoint's tensor keys.
    
    This is more reliable than filename patterns for unusual checkpoints.
    
    Returns:
        "sdxl", "zimg", "lumina", or None if unknown
    """
    if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith('.safetensors'):
        return None
    
    try:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            # Check for Z-Image / Lumina-Next signature keys
            has_cap_embedder = any('cap_embedder' in k for k in keys)
            has_context_refiner = any('context_refiner' in k for k in keys)
            has_dit_blocks = any('double_layers' in k or 'single_layers' in k for k in keys)
            
            if has_cap_embedder or has_context_refiner or has_dit_blocks:
                logger.info("Detected Lumina/Z-Image architecture from checkpoint keys")
                return "zimg"
            
            # Check for SDXL UNet signature keys
            has_label_emb = any('label_emb' in k for k in keys)
            has_input_blocks = any('input_blocks' in k for k in keys)
            has_middle_block = any('middle_block' in k for k in keys)
            has_output_blocks = any('output_blocks' in k for k in keys)
            
            if (has_label_emb or has_input_blocks) and has_middle_block and has_output_blocks:
                logger.debug("Detected SDXL UNet architecture from checkpoint keys")
                return "sdxl"
            
            # Check for diffusion_model prefix (common in both but helps narrow down)
            has_diffusion_model = any('diffusion_model' in k for k in keys)
            if has_diffusion_model:
                # If it has diffusion_model but not SDXL structure, might be other format
                if has_cap_embedder or has_context_refiner:
                    return "zimg"
                    
    except Exception as e:
        logger.debug(f"Could not inspect checkpoint keys: {e}")
    
    return None


def is_zimage_checkpoint(checkpoint: str) -> bool:
    """Check if checkpoint is a Z-Image model."""
    if not checkpoint:
        return False
    filename = os.path.basename(checkpoint).lower()
    for pattern in ZIMAGE_MODEL_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False


def is_anima_checkpoint(checkpoint: str) -> bool:
    """Check if checkpoint is an Anima model (not currently supported)."""
    if not checkpoint:
        return False
    filename = os.path.basename(checkpoint).lower()
    for pattern in ANIMA_MODEL_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    if "circlestone-labs/anima" in checkpoint.lower():
        return True
    return False


def detect_prediction_type(checkpoint: str) -> Tuple[str, str]:
    """
    Detect prediction type from checkpoint filename and/or metadata.
    
    Priority:
    1. Safetensors metadata (most reliable)
    2. Filename pattern matching
    3. Default to epsilon
    
    Returns:
        Tuple of (prediction_type, config_source)
    """
    if not checkpoint:
        return "epsilon", "default"
    
    filename = os.path.basename(checkpoint).lower()
    
    # First, try to read safetensors metadata (most reliable)
    full_path = checkpoint
    if os.path.isfile(full_path) and full_path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(full_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata:
                    # Check modelspec.prediction_type
                    if 'modelspec.prediction_type' in metadata:
                        pred = metadata['modelspec.prediction_type']
                        if pred in ('epsilon', 'v_prediction'):
                            logger.info(f"Found prediction_type in metadata: {pred}")
                            return pred, "safetensors_metadata"
                    # Check ss_v2 (kohya training flag)
                    if metadata.get('ss_v2') == 'True':
                        logger.info("Found ss_v2=True in metadata -> v_prediction")
                        return 'v_prediction', "safetensors_ss_v2"
        except Exception as e:
            logger.debug(f"Could not read safetensors metadata: {e}")
    
    # Fall back to filename pattern matching
    for pattern, pred_type in KNOWN_MODEL_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            logger.debug(f"Matched pattern '{pattern}' -> {pred_type}")
            return pred_type, f"pattern:{pattern}"
    
    # Default to epsilon
    return "epsilon", "default"


def detect_checkpoint_config(checkpoint: str, checkpoint_dir: Optional[str] = None) -> CheckpointConfig:
    """
    Detect full checkpoint configuration.
    
    Detection priority:
    1. Content-based detection (inspect checkpoint keys) - most reliable
    2. Filename pattern matching
    3. Default to SDXL with epsilon
    
    Args:
        checkpoint: Checkpoint filename or path
        checkpoint_dir: Optional directory to look for checkpoint (for metadata reading)
        
    Returns:
        CheckpointConfig with detected settings
    """
    # Try to get full path for content-based detection
    full_path = checkpoint
    if not os.path.isabs(checkpoint) and checkpoint_dir:
        potential_path = os.path.join(checkpoint_dir, checkpoint)
        if os.path.isfile(potential_path):
            full_path = potential_path
    
    # Priority 1: Content-based detection (most reliable)
    if os.path.isfile(full_path):
        model_type_from_keys = detect_model_type_from_keys(full_path)
        if model_type_from_keys == "zimg":
            logger.info(f"Detected Z-Image model from checkpoint structure: {os.path.basename(checkpoint)}")
            return CheckpointConfig(
                model_type="zimg",
                prediction_type="epsilon",
                is_zimage=True,
                is_anima=False,
                config_source="checkpoint_keys"
            )
    
    # Priority 2: Filename pattern matching
    if is_zimage_checkpoint(checkpoint):
        logger.info(f"Detected Z-Image model from filename: {checkpoint}")
        return CheckpointConfig(
            model_type="zimg",
            prediction_type="epsilon",
            is_zimage=True,
            is_anima=False,
            config_source="zimage_pattern"
        )
    
    if is_anima_checkpoint(checkpoint):
        logger.warning(f"Anima model detected but not supported: {checkpoint}")
        return CheckpointConfig(
            model_type="anima",
            prediction_type="epsilon",
            is_zimage=False,
            is_anima=True,
            config_source="anima_pattern"
        )
    
    # Default: SDXL - detect prediction type (pass full path for metadata)
    pred_type, source = detect_prediction_type(full_path)
    
    logger.info(f"Detected SDXL model: {os.path.basename(checkpoint)} (prediction_type={pred_type}, source={source})")
    return CheckpointConfig(
        model_type="sdxl",
        prediction_type=pred_type,
        is_zimage=False,
        is_anima=False,
        config_source=source
    )


def get_model_type_for_checkpoint(checkpoint: Optional[str], checkpoint_dir: Optional[str] = None) -> str:
    """
    Get the pipeline type to use for a checkpoint.
    
    Args:
        checkpoint: Checkpoint filename or None for default
        checkpoint_dir: Optional checkpoint directory for content-based detection
        
    Returns:
        "sdxl" or "zimg"
    """
    if not checkpoint:
        return "sdxl"
    
    # If no checkpoint_dir provided, try to get from config
    if checkpoint_dir is None:
        try:
            from .config import get_config
            checkpoint_dir = get_config().sdxl_checkpoints_dir
        except Exception:
            pass
    
    config = detect_checkpoint_config(checkpoint, checkpoint_dir)
    
    if config.is_anima:
        logger.warning("Anima models not supported, falling back to SDXL")
        return "sdxl"
    
    return config.model_type
