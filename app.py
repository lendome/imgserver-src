"""Flask application factory."""

import logging
from flask import Flask, jsonify
from flask_cors import CORS

from .routes import get_all_blueprints, classify_root_bp, tts_bp, ltx_bp
from .pipelines import get_loaded_pipelines, unload_all
from .vram import vram_manager
from .gpu_lock import gpu_lock
from .server_state import server_state
from .classifiers import classifier_registry
from .queue import get_queue, QueueWorker

logger = logging.getLogger(__name__)

# Global worker reference
_queue_worker = None


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for all origins
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Add CORS headers to all responses
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Register all blueprints dynamically
    for bp in get_all_blueprints():
        # Skip blueprints that need special handling
        if bp.name in ('classify_root', 'tts', 'ltx'):
            continue
        app.register_blueprint(bp)
    
    # Register blueprints with special url_prefix handling
    app.register_blueprint(classify_root_bp)  # Direct /nsfw, /aesthetic routes
    app.register_blueprint(tts_bp, url_prefix='/tts')
    app.register_blueprint(ltx_bp, url_prefix='/ltx')
    
    # Health/status endpoints
    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/status")
    def status():
        loaded = get_loaded_pipelines()
        response = {
            "loaded_pipelines": list(loaded.keys()),
            "classifiers": {
                "current": classifier_registry.current_name,
                "available": classifier_registry.available,
            },
            "vram": {
                "free_gb": round(vram_manager.get_free() / (1024**3), 2),
                "used_gb": round(vram_manager.get_used() / (1024**3), 2),
                "total_gb": round(vram_manager.get_total() / (1024**3), 2),
            },
            "uptime_seconds": round(server_state.uptime, 2),
            "generation_count": server_state.generation_count,
            "error_count": server_state.error_count,
            "idle_seconds": round(server_state.idle_time, 2),
            "gpu_busy": gpu_lock.is_busy(),
            "gpu_holder": gpu_lock.current_holder,
            "video_active": gpu_lock.is_video_active(),
        }
        if server_state.current_task:
            response["current_task"] = server_state.current_task
        return jsonify(response)
    
    @app.route("/unload", methods=["POST"])
    def unload():
        count = unload_all()
        return jsonify({"unloaded": count})
    
    # Error handlers
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.exception("Unhandled exception")
        return jsonify({"error": str(e)}), 500
    
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Not found"}), 404
    
    logger.info("App created with routes: /sdxl/generate, /zimg/generate, /tts/generate, /ltx/generate, /status, /health, /unload")
    logger.info("Queue API routes: /sdxl/queue, /job/<job_id>, /queue/status")
    
    # Start queue worker
    start_worker()
    
    return app


def _queue_generate_fn(request_data: dict):
    """
    Generation function called by the queue worker.
    
    Similar to /sdxl/generate but returns base64 image data instead of Response.
    """
    import base64
    import random
    
    from .pipelines import get_pipeline
    from .tipo import get_tipo_engine
    from .lora import LoRAManager
    from .embeddings import EmbeddingManager
    from .recovery import safe_generate
    from .checkpoint_detect import get_model_type_for_checkpoint
    
    # Core parameters
    prompt = request_data.get("prompt")
    negative_prompt = request_data.get("negative_prompt", "")
    width = request_data.get("width")
    height = request_data.get("height")
    steps = request_data.get("steps")
    cfg_scale = request_data.get("cfg_scale") or request_data.get("guidance_scale")
    seed = request_data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Model selection
    checkpoint = request_data.get("checkpoint")
    pipeline_type = request_data.get("pipeline")
    
    # Auto-detect pipeline type from checkpoint if not specified
    if not pipeline_type and checkpoint:
        pipeline_type = get_model_type_for_checkpoint(checkpoint)
    elif not pipeline_type:
        pipeline_type = "sdxl"
    
    # Check for unsupported pipeline types in queue
    if pipeline_type == "ltx":
        logger.warning("LTX video generation via queue is not supported yet - video generation may fail")
    
    
    # Scheduler options
    scheduler = request_data.get("scheduler")
    use_karras = request_data.get("use_karras", False) or request_data.get("karras", False)
    prediction_type = request_data.get("prediction_type")
    
    # TIPO
    use_tipo = request_data.get("use_tipo", False)
    tipo_max_tokens = request_data.get("tipo_max_tokens", 256)
    tipo_prepend_quality = request_data.get("tipo_prepend_quality", True)
    
    # Hires fix
    use_hires_fix = request_data.get("use_hires_fix", False)
    upscale_by = request_data.get("upscale_by", 1.5)
    hires_denoising = request_data.get("hires_denoising")
    
    # Monochrome
    monochrome = request_data.get("monochrome", False)
    
    # Down factor
    down = request_data.get("down", 1.0)
    if down is not None and down < 1.0:
        down = float(down)
        if width:
            width = round(width * down / 16) * 16
        if height:
            height = round(height * down / 16) * 16
    
    # LoRAs and embeddings
    loras = request_data.get("loras", [])
    embeddings = request_data.get("embeddings", [])
    
    # Safety
    validate_output = request_data.get("validate_output", True)
    
    # TIPO Enhancement
    if use_tipo and pipeline_type == "sdxl":
        try:
            tipo = get_tipo_engine()
            prompt = tipo.enhance(
                prompt,
                max_tokens=tipo_max_tokens,
                prepend_quality=tipo_prepend_quality
            )
            logger.info(f"TIPO enhanced prompt: {prompt[:100]}...")
        except Exception as e:
            logger.warning(f"TIPO failed, using original prompt: {e}")
    
    # Get Pipeline
    if pipeline_type == "zimg":
        pipeline = get_pipeline("zimg")
    else:
        pipeline = get_pipeline("sdxl", checkpoint=checkpoint)
    
    if not pipeline.is_loaded:
        pipeline.load()
    
    # Load LoRAs
    if loras and pipeline_type == "sdxl":
        try:
            lora_manager = LoRAManager()
            lora_manager.load_loras(pipeline._pipe, loras)
        except Exception as e:
            logger.warning(f"LoRA loading failed: {e}")
    
    # Load Embeddings
    if embeddings and pipeline_type == "sdxl":
        try:
            emb_manager = EmbeddingManager()
            emb_manager.load_embeddings(pipeline._pipe, embeddings)
        except Exception as e:
            logger.warning(f"Embedding loading failed: {e}")
    
    # Build generate kwargs
    generate_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }
    
    if width:
        generate_kwargs["width"] = width
    if height:
        generate_kwargs["height"] = height
    if steps:
        generate_kwargs["steps"] = steps
    if cfg_scale:
        generate_kwargs["cfg_scale"] = cfg_scale
    
    # SDXL-specific options
    if pipeline_type == "sdxl":
        if scheduler:
            generate_kwargs["scheduler"] = scheduler
        generate_kwargs["use_karras"] = use_karras
        if prediction_type:
            generate_kwargs["prediction_type"] = prediction_type
        if use_hires_fix:
            generate_kwargs["use_hires_fix"] = True
            generate_kwargs["upscale_by"] = upscale_by
            if hires_denoising is not None:
                generate_kwargs["hires_denoising"] = hires_denoising
        if monochrome:
            generate_kwargs["monochrome"] = True
    
    # Generate
    if validate_output:
        image = safe_generate(pipeline, generate_kwargs)
    else:
        image = pipeline.generate(**generate_kwargs)
    
    # Return as base64
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "seed": seed,
        "width": image.width,
        "height": image.height,
    }


def start_worker():
    """Start the queue worker thread."""
    global _queue_worker
    
    if _queue_worker is not None:
        logger.info("Queue worker already started")
        return
    
    queue = get_queue()
    _queue_worker = QueueWorker(queue, _queue_generate_fn, name="SDXLQueueWorker")
    _queue_worker.start()
    logger.info("Queue worker started")
