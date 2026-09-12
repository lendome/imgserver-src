"""Checkpoint management routes."""

import logging
import hashlib
from pathlib import Path
from typing import Optional

from flask import Blueprint, request, jsonify
import requests

from ..config import get_config
from ..pipelines import get_pipeline, unload_pipeline

logger = logging.getLogger(__name__)

checkpoint_bp = Blueprint("checkpoint", __name__)


@checkpoint_bp.route("/checkpoints", methods=["GET"])
def list_checkpoints():
    """
    List all available checkpoints.
    
    Returns:
        JSON with available checkpoints and current state.
    """
    config = get_config()
    checkpoints = config.get_available_checkpoints()
    
    return jsonify({
        "checkpoints": checkpoints,
        "current": config.default_checkpoint,
        "directory": config.sdxl_checkpoints_dir,
    })


@checkpoint_bp.route("/checkpoint/switch", methods=["POST"])
def switch_checkpoint():
    """
    Switch to a different checkpoint.
    
    JSON body:
        checkpoint (str): Required. Checkpoint filename.
        prediction_type (str): Optional. epsilon, v_prediction, sample.
    
    Returns:
        JSON with success status and current checkpoint.
    """
    try:
        data = request.get_json() or {}
        checkpoint = data.get("checkpoint")
        
        if not checkpoint:
            return jsonify({"error": "checkpoint is required"}), 400
        
        config = get_config()
        checkpoint_path = config.get_checkpoint_path(checkpoint)
        
        # Verify checkpoint exists
        if not checkpoint_path.exists():
            available = config.get_available_checkpoints()
            return jsonify({
                "error": f"Checkpoint not found: {checkpoint}",
                "available": available,
            }), 404
        
        # Unload current SDXL pipeline to switch checkpoint
        unload_pipeline("sdxl")
        
        # Update default checkpoint in config (runtime only)
        config.default_checkpoint = checkpoint
        
        # Optionally pre-load the new checkpoint
        preload = data.get("preload", False)
        if preload:
            pipeline = get_pipeline("sdxl", checkpoint=checkpoint)
            pipeline.load()
        
        logger.info(f"Switched to checkpoint: {checkpoint}")
        
        return jsonify({
            "success": True,
            "checkpoint": checkpoint,
            "preloaded": preload,
        })
        
    except Exception as e:
        logger.exception("Checkpoint switch failed")
        return jsonify({"error": str(e)}), 500


@checkpoint_bp.route("/checkpoint/reload", methods=["POST"])
def reload_checkpoint():
    """
    Force reload the current checkpoint.
    
    Unloads and reloads the SDXL pipeline from disk.
    
    Returns:
        JSON with success status.
    """
    try:
        config = get_config()
        
        # Unload current pipeline
        unload_pipeline("sdxl")
        
        # Reload
        pipeline = get_pipeline("sdxl")
        pipeline.load()
        
        logger.info(f"Reloaded checkpoint: {config.default_checkpoint}")
        
        return jsonify({
            "success": True,
            "checkpoint": config.default_checkpoint,
        })
        
    except Exception as e:
        logger.exception("Checkpoint reload failed")
        return jsonify({"error": str(e)}), 500


@checkpoint_bp.route("/checkpoint/download", methods=["POST"])
def download_checkpoint():
    """
    Download a checkpoint from URL.
    
    JSON body:
        url (str): Required. URL to download from.
        filename (str): Optional. Filename to save as.
    
    Returns:
        JSON with download status.
    """
    try:
        data = request.get_json() or {}
        url = data.get("url")
        
        if not url:
            return jsonify({"error": "url is required"}), 400
        
        config = get_config()
        
        # Generate filename if not provided
        filename = data.get("filename")
        if not filename:
            # Extract from URL or use hash
            if "/" in url:
                filename = url.rsplit("/", 1)[-1]
            else:
                filename = hashlib.md5(url.encode()).hexdigest()[:16] + ".safetensors"
        
        # Ensure valid extension
        if not any(filename.endswith(ext) for ext in [".safetensors", ".ckpt", ".pt"]):
            filename += ".safetensors"
        
        checkpoint_path = Path(config.sdxl_checkpoints_dir) / filename
        
        # Check if already exists
        if checkpoint_path.exists():
            return jsonify({
                "error": f"File already exists: {filename}",
                "path": str(checkpoint_path),
            }), 409
        
        # Download
        logger.info(f"Downloading checkpoint from {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        # Write to temp file then rename (atomic)
        temp_path = checkpoint_path.with_suffix(".downloading")
        downloaded = 0
        
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if int(pct) % 10 == 0:
                        logger.info(f"Download progress: {pct:.0f}%")
        
        # Rename to final
        temp_path.rename(checkpoint_path)
        
        size_gb = checkpoint_path.stat().st_size / (1024 ** 3)
        logger.info(f"Downloaded checkpoint: {filename} ({size_gb:.2f} GB)")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "size_gb": round(size_gb, 2),
            "path": str(checkpoint_path),
        })
        
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return jsonify({"error": f"Download failed: {e}"}), 500
    except Exception as e:
        logger.exception("Checkpoint download failed")
        return jsonify({"error": str(e)}), 500
