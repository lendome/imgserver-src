"""SDXL image generation routes."""

import logging
from io import BytesIO

from flask import Blueprint, request, jsonify, Response
from PIL import Image

from ..pipelines import get_pipeline
from ..abort import abort_controller, GenerationAbortedError
from ..gpu_lock import gpu_lock, VideoGenerationActiveError

logger = logging.getLogger(__name__)

sdxl_bp = Blueprint("sdxl", __name__, url_prefix="/sdxl")


@sdxl_bp.route("/generate", methods=["POST"])
def generate():
    """Generate image using SDXL pipeline.
    
    JSON body:
        prompt (str): Required. The text prompt for generation.
        negative_prompt (str): Optional. Negative prompt.
        width (int): Optional. Image width.
        height (int): Optional. Image height.
        steps (int): Optional. Number of inference steps.
        cfg_scale (float): Optional. Classifier-free guidance scale.
        seed (int): Optional. Random seed for reproducibility.
        scheduler (str): Optional. Scheduler name (euler, euler_a, dpm_2m, dpm_2m_karras).
        checkpoint (str): Optional. Checkpoint file to use.
        monochrome (bool): Optional. Convert output to black and white. Default: False.
    
    Returns:
        Raw PNG image bytes with image/png mimetype.
    """
    try:
        # Check if video generation is active
        if gpu_lock.is_video_active():
            return jsonify({
                "error": "Video generation in progress",
                "retry_after": 60
            }), 503
        
        data = request.get_json() or {}
        
        # Validate required fields
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        # Extract optional parameters
        checkpoint = data.get("checkpoint")
        negative_prompt = data.get("negative_prompt", "")
        width = data.get("width")
        height = data.get("height")
        steps = data.get("steps")
        cfg_scale = data.get("cfg_scale")
        seed = data.get("seed")
        scheduler = data.get("scheduler")
        monochrome = data.get("monochrome", False)
        
        # Get pipeline and ensure loaded
        pipeline = get_pipeline("sdxl", checkpoint=checkpoint)
        pipeline.load()
        
        # Generate image with abort tracking
        abort_controller.start_generation()
        try:
            image = pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                scheduler=scheduler,
                monochrome=monochrome,
            )
        finally:
            abort_controller.end_generation()
        
        # Return raw PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return Response(buffer.getvalue(), mimetype="image/png")
    
    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"SDXL generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499
        
    except Exception as e:
        logger.exception("SDXL generation failed")
        return jsonify({"error": str(e)}), 500


@sdxl_bp.route("/img2img", methods=["POST"])
def img2img():
    """Refine an existing image using SDXL img2img.

    Accepts the input image as raw bytes in the request body (any format
    PIL can open: PNG, JPEG, WebP, ...), with parameters via query string,
    or as multipart/form-data with the image in an "image" file field and
    parameters as form fields.

    Parameters:
        prompt (str): Required. The text prompt for refinement.
        negative_prompt (str): Optional. Negative prompt.
        strength (float): Optional. Denoising strength 0.0-1.0. Default: 0.75.
        steps (int): Optional. Number of inference steps.
        cfg_scale (float): Optional. Classifier-free guidance scale.
        seed (int): Optional. Random seed for reproducibility.
        scheduler (str): Optional. Scheduler name (euler, euler_a, dpm_2m, dpm_2m_karras).
        checkpoint (str): Optional. Checkpoint file to use.
        monochrome (bool): Optional. Convert output to black and white. Default: False.

    Returns:
        Raw PNG image bytes with image/png mimetype.
    """
    try:
        # Check if video generation is active
        if gpu_lock.is_video_active():
            return jsonify({
                "error": "Video generation in progress",
                "retry_after": 60
            }), 503

        # Extract input image bytes (multipart file field or raw body)
        if "image" in request.files:
            image_bytes = request.files["image"].read()
            params = request.form
        else:
            image_bytes = request.get_data()
            params = request.args

        if not image_bytes:
            return jsonify({"error": "image bytes are required in the request body"}), 400

        try:
            init_image = Image.open(BytesIO(image_bytes))
            init_image.load()
        except Exception:
            return jsonify({"error": "request body is not a valid image"}), 400

        # Validate required fields
        prompt = params.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # Extract optional parameters
        checkpoint = params.get("checkpoint")
        negative_prompt = params.get("negative_prompt", "")
        strength = params.get("strength")
        steps = params.get("steps")
        cfg_scale = params.get("cfg_scale")
        seed = params.get("seed")
        scheduler = params.get("scheduler")
        monochrome = params.get("monochrome", "false").lower() in ("true", "1", "yes")

        # Get pipeline and ensure loaded
        pipeline = get_pipeline("sdxl", checkpoint=checkpoint)
        pipeline.load()

        # Run img2img with abort tracking
        abort_controller.start_generation()
        try:
            image = pipeline.img2img(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=float(strength) if strength is not None else 0.75,
                steps=int(steps) if steps is not None else None,
                cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
                seed=int(seed) if seed is not None else None,
                scheduler=scheduler,
                monochrome=monochrome,
            )
        finally:
            abort_controller.end_generation()

        # Return raw PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return Response(buffer.getvalue(), mimetype="image/png")

    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"SDXL img2img aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.exception("SDXL img2img failed")
        return jsonify({"error": str(e)}), 500
