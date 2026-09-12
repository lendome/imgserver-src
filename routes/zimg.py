"""Z-Image generation routes."""

import logging
from flask import Blueprint, request, jsonify, Response
from io import BytesIO

from ..pipelines import get_pipeline
from ..abort import abort_controller, GenerationAbortedError
from ..gpu_lock import gpu_lock, VideoGenerationActiveError

logger = logging.getLogger(__name__)

zimg_bp = Blueprint("zimg", __name__, url_prefix="/zimg")


@zimg_bp.route("/generate", methods=["POST"])
def generate():
    """Generate an image using the zimg pipeline."""
    try:
        # Check if video generation is active
        if gpu_lock.is_video_active():
            return jsonify({
                "error": "Video generation in progress",
                "retry_after": 60
            }), 503
        
        data = request.get_json() or {}
        
        # Validate required field
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        # Optional parameters with defaults
        width = data.get("width", 512)
        height = data.get("height", 512)
        steps = data.get("steps", 20)
        seed = data.get("seed")
        
        # Get pipeline and generate with abort tracking
        pipeline = get_pipeline("zimg")
        
        abort_controller.start_generation()
        try:
            image = pipeline.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
        finally:
            abort_controller.end_generation()
        
        # Return raw PNG image
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return Response(buffer.getvalue(), mimetype="image/png")
    
    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"ZImg generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499
        
    except Exception as e:
        logger.exception("ZImg generation failed")
        return jsonify({"error": str(e)}), 500
