"""LTX video generation routes."""

import logging
import os
import tempfile
import time

import requests
from flask import Blueprint, request, jsonify, Response
from PIL import Image

from ..pipelines import get_pipeline
from ..gpu_lock import gpu_lock, ClientDisconnectedWhileWaiting
from ..abort import abort_controller, GenerationAbortedError
from ..configs.ltx_presets import get_preset, LTXPreset, PRESETS
from ..pipelines.ltx import LTXGuiderParams

logger = logging.getLogger(__name__)

ltx_bp = Blueprint("ltx", __name__, url_prefix="/ltx")


@ltx_bp.route("/generate", methods=["POST"])
def ltx_generate():
    """
    Generate video from text prompt using LTX pipeline.
    
    JSON body:
        prompt (str): Required. Text prompt for video generation.
        negative_prompt (str): Optional. Negative prompt (default "").
        image_url (str): Optional. URL to source image for image-to-video.
        width (int): Optional. Video width (default from pipeline).
        height (int): Optional. Video height (default from pipeline).
        num_frames (int): Optional. Number of frames to generate (default 41).
        steps (int): Optional. Inference steps (default 50).
        guidance_scale (float): Optional. Guidance scale (default 5.0).
        seed (int): Optional. Random seed for reproducibility.
        target_resolution (int): Optional. Target average resolution for scaling (default 768).
        preset (str): Optional. Preset name ("quality", "balanced", "fast", "memory_efficient").
        cfg_scale (float): Optional. CFG scale override (default from preset or 3.0).
        stg_scale (float): Optional. STG scale override (default 0.0).
        rescale_scale (float): Optional. Rescale override (default 0.0).
        use_gradient_estimation (bool): Optional. Fast inference mode (default False).
        ge_gamma (float): Optional. Gradient estimation gamma (default 2.0).
        img_compression (int): Optional. Image preprocessing compression (0-100, default 42).
                               Higher = more compression. Helps prevent "still video" output.
        add_motion_negative (bool): Optional. Add motion negative prompt (default True).
                                    Appends "still image, still video, no motion" to negative prompt.
    
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
        
        # Apply preset if specified
        preset_name = data.get("preset")
        preset = None
        if preset_name:
            try:
                preset = get_preset(preset_name)
                # Use preset values as defaults
                logger.info(f"Using LTX preset: {preset_name}")
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        
        # Optional parameters (preset provides defaults if available)
        negative_prompt = data.get("negative_prompt", "")
        image_url = data.get("image_url")
        width = data.get("width")
        height = data.get("height")
        num_frames = data.get("num_frames", 41)
        steps = data.get("steps", preset.num_inference_steps if preset else 50)
        guidance_scale = data.get("guidance_scale", preset.guidance_scale if preset else 5.0)
        seed = data.get("seed")
        fps = data.get("fps", 24)  # Output video framerate (default 24)
        loras = data.get("loras")  # List of {"url": "...", "strength": 1.0}
        target_resolution = data.get("target_resolution", 768)  # Target average resolution for scaling
        img_compression = data.get("img_compression", 42)  # Image preprocessing compression (0-100)
        add_motion_negative = data.get("add_motion_negative", True)  # Add motion negative prompt
        
        # Build guider params
        guider_params = None
        cfg_scale = data.get("cfg_scale")
        stg_scale = data.get("stg_scale")
        rescale_scale = data.get("rescale_scale")
        
        if any([cfg_scale, stg_scale, rescale_scale]) or (preset and preset.stg_scale > 0):
            guider_params = LTXGuiderParams(
                cfg_scale=cfg_scale or (preset.guidance_scale if preset else 3.0),
                stg_scale=stg_scale if stg_scale is not None else (preset.stg_scale if preset else 0.0),
                rescale_scale=rescale_scale if rescale_scale is not None else (preset.rescale_scale if preset else 0.0),
            )
        
        # Acquire GPU lock and set video generation active
        with gpu_lock.acquire("ltx_video"):
            gpu_lock.set_video_active(True)
            try:
                # Get LTX pipeline
                pipeline = get_pipeline("ltx")
                
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
                    if loras:
                        gen_kwargs["loras"] = loras
                    if target_resolution != 768:
                        gen_kwargs["target_resolution"] = target_resolution
                    if guider_params:
                        gen_kwargs["guider_params"] = guider_params
                    if data.get("use_gradient_estimation") or (preset and preset.use_gradient_estimation):
                        gen_kwargs["use_gradient_estimation"] = True
                        gen_kwargs["ge_gamma"] = data.get("ge_gamma", preset.ge_gamma if preset else 2.0)
                    
                    # Add motion-related parameters (critical for preventing "still video")
                    gen_kwargs["img_compression"] = img_compression
                    gen_kwargs["add_motion_negative"] = add_motion_negative
                    
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
                        
                        logger.info(f"LTX generated: '{prompt}' -> {len(video_bytes)} bytes in {gen_time:.2f}s")
                        
                        # Return MP4 with metadata headers
                        resp = Response(video_bytes, mimetype="video/mp4")
                        resp.headers["X-LTX-Model"] = "ltx"
                        resp.headers["X-LTX-Gen-Seconds"] = f"{gen_time:.3f}"
                        resp.headers["X-LTX-Frames"] = str(num_frames)
                        resp.headers["X-LTX-FPS"] = str(fps)
                        
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
        logger.info(f"LTX generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499
    except ClientDisconnectedWhileWaiting:
        logger.info("Client disconnected while waiting for GPU")
        return jsonify({"error": "Client disconnected"}), 499
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        abort_controller.abort()
        logger.warning(f"Client disconnected during LTX generation: {e}")
        return jsonify({"error": "Client disconnected"}), 499
    except Exception as e:
        logger.exception("LTX generation failed")
        return jsonify({"error": str(e)}), 500


@ltx_bp.route("/presets", methods=["GET"])
def ltx_presets():
    """List available LTX generation presets."""
    return jsonify({
        name: {
            "description": p.description,
            "steps": p.num_inference_steps,
            "guidance_scale": p.guidance_scale,
        }
        for name, p in PRESETS.items()
    })


@ltx_bp.route("/status", methods=["GET"])
def ltx_status():
    """Get LTX pipeline status."""
    try:
        from ..pipelines import get_loaded_pipelines
        
        loaded_pipelines = get_loaded_pipelines()
        
        # Check if LTX is loaded
        ltx_loaded = "ltx" in loaded_pipelines
        ltx_device = "cpu"
        
        if ltx_loaded:
            pipeline = loaded_pipelines["ltx"]
            ltx_device = pipeline._device if hasattr(pipeline, '_device') else "cuda"
        
        return jsonify({
            "loaded": ltx_loaded,
            "model": "ltx",
            "device": ltx_device
        })
    except Exception as e:
        logger.exception("LTX status check failed")
        return jsonify({
            "loaded": False,
            "model": "ltx",
            "device": "unknown",
            "error": str(e)
        })
