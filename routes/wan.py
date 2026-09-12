"""Wan 2.1 video generation routes."""

import logging
import os
import tempfile
import time

import requests
from flask import Blueprint, request, jsonify, Response
from PIL import Image

from ..pipelines import get_pipeline
from ..gpu_lock import gpu_lock
from ..abort import abort_controller, GenerationAbortedError
from ..gpu_lock import ClientDisconnectedWhileWaiting

logger = logging.getLogger(__name__)

wan_bp = Blueprint("wan", __name__, url_prefix="/wan")


@wan_bp.route("/generate", methods=["POST"])
def wan_generate():
    """
    Generate video from text prompt using Wan 2.1 pipeline.
    
    JSON body:
        prompt (str): Required. Text prompt for video generation.
        negative_prompt (str): Optional. Negative prompt (default uses Wan defaults).
        image_url (str): Optional. URL to source image for image-to-video.
        width (int): Optional. Video width (auto-calculated if not provided).
        height (int): Optional. Video height (auto-calculated if not provided).
        num_frames (int): Optional. Number of frames to generate (default 81).
        steps (int): Optional. Inference steps (default 50).
        guidance_scale (float): Optional. Guidance scale (default 5.0).
        seed (int): Optional. Random seed for reproducibility.
        resolution (str): Optional. Target resolution - "480p" or "720p" (default "480p").
        fps (int): Optional. Output video framerate (default 16).
    
    Returns:
        MP4 video bytes with metadata headers.
    """
    temp_file = None
    
    try:
        data = request.get_json() or {}
        
        # Validate required fields
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        # Optional parameters
        negative_prompt = data.get("negative_prompt", "")
        image_url = data.get("image_url")
        width = data.get("width")
        height = data.get("height")
        num_frames = data.get("num_frames", 81)
        steps = data.get("steps", 50)
        guidance_scale = data.get("guidance_scale", 5.0)
        seed = data.get("seed")
        fps = data.get("fps", 16)  # Wan default is 16
        resolution = data.get("resolution", "480p")  # "480p" or "720p"
        
        # Acquire GPU lock and set video generation active
        with gpu_lock.acquire("wan_video"):
            gpu_lock.set_video_active(True)
            try:
                # Get Wan pipeline
                pipeline = get_pipeline("wan")
                
                # Update pipeline resolution/mode if needed
                if hasattr(pipeline, '_resolution') and pipeline._resolution != resolution:
                    logger.info(f"Resolution changed from {pipeline._resolution} to {resolution}, reloading...")
                    pipeline.unload()
                    pipeline._resolution = resolution
                
                # Ensure loaded
                if not pipeline.is_loaded:
                    pipeline.load()
                
                # Handle image URL - download to temp file if provided
                image_path = None
                source_image = None
                
                try:
                    if image_url:
                        logger.info(f"Downloading source image from: {image_url}")
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()
                        
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        temp_file.write(response.content)
                        temp_file.close()
                        image_path = temp_file.name
                        logger.debug(f"Source image saved to: {image_path}")
                        
                        # Load as PIL Image
                        source_image = Image.open(image_path)
                        # Convert to RGB if needed (handles RGBA, etc.)
                        if source_image.mode != 'RGB':
                            source_image = source_image.convert('RGB')
                    
                    # Generate video
                    start_time = time.time()
                    
                    # Build generation kwargs
                    gen_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_frames": num_frames,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                    }
                    
                    # Add optional parameters if provided
                    if width is not None:
                        gen_kwargs["width"] = width
                    if height is not None:
                        gen_kwargs["height"] = height
                    if seed is not None:
                        gen_kwargs["seed"] = seed
                    if source_image is not None:
                        gen_kwargs["image"] = source_image
                    
                    # Generate video frames with abort tracking
                    abort_controller.start_generation()
                    try:
                        frames = pipeline.generate(**gen_kwargs)
                    finally:
                        abort_controller.end_generation()
                    
                    gen_time = time.time() - start_time
                    
                    # Convert frames to MP4 video using diffusers export_to_video
                    from diffusers.utils import export_to_video
                    
                    # Create temporary video file
                    video_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                    video_path = video_temp.name
                    video_temp.close()
                    
                    try:
                        # Export frames to video
                        export_to_video(
                            frames,
                            video_path,
                            fps=fps
                        )
                        
                        # Read video bytes
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        logger.info(f"Wan generated: '{prompt}' -> {len(video_bytes)} bytes in {gen_time:.2f}s")
                        
                        # Return MP4 with metadata headers
                        resp = Response(video_bytes, mimetype="video/mp4")
                        resp.headers["X-Wan-Model"] = "wan"
                        resp.headers["X-Wan-Gen-Seconds"] = f"{gen_time:.3f}"
                        resp.headers["X-Wan-Frames"] = str(num_frames)
                        resp.headers["X-Wan-FPS"] = str(fps)
                        resp.headers["X-Wan-Resolution"] = resolution
                        
                        return resp
                        
                    finally:
                        # Clean up video temp file
                        if os.path.exists(video_path):
                            try:
                                os.unlink(video_path)
                            except Exception as e:
                                logger.warning(f"Failed to delete video temp file: {e}")
                
                finally:
                    # Clean up image temp file
                    if image_path and os.path.exists(image_path):
                        try:
                            os.unlink(image_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete image temp file: {e}")
            finally:
                # Always clear video active flag when done
                gpu_lock.set_video_active(False)
    
    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"Wan generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499
    except ClientDisconnectedWhileWaiting:
        logger.info("Client disconnected while waiting for GPU")
        return jsonify({"error": "Client disconnected"}), 499
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        abort_controller.abort()
        logger.warning(f"Client disconnected during Wan generation: {e}")
        return jsonify({"error": "Client disconnected"}), 499
    except Exception as e:
        logger.exception("Wan generation failed")
        return jsonify({"error": str(e)}), 500


@wan_bp.route("/status", methods=["GET"])
def wan_status():
    """Get Wan pipeline status."""
    try:
        from ..pipelines import get_loaded_pipelines
        
        loaded_pipelines = get_loaded_pipelines()
        
        # Check if Wan is loaded
        wan_loaded = "wan" in loaded_pipelines
        wan_device = "cpu"
        wan_mode = "unknown"
        wan_resolution = "unknown"
        
        if wan_loaded:
            pipeline = loaded_pipelines["wan"]
            wan_device = pipeline._device if hasattr(pipeline, '_device') else "cuda"
            wan_mode = pipeline._mode if hasattr(pipeline, '_mode') else "unknown"
            wan_resolution = pipeline._resolution if hasattr(pipeline, '_resolution') else "unknown"
        
        return jsonify({
            "loaded": wan_loaded,
            "model": "wan",
            "device": wan_device,
            "mode": wan_mode,
            "resolution": wan_resolution
        })
    except Exception as e:
        logger.exception("Wan status check failed")
        return jsonify({
            "loaded": False,
            "model": "wan",
            "device": "unknown",
            "error": str(e)
        })
