"""Unified generation route with all features."""

import base64
import logging
import os
import random
import urllib.request
import uuid
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

from flask import Blueprint, request, jsonify, Response
from PIL import Image

from ..pipelines import get_pipeline
from ..abort import abort_controller, GenerationAbortedError
from ..tipo import get_tipo_engine
from ..lora import LoRAManager
from ..embeddings import EmbeddingManager
from ..recovery import safe_generate
from ..validation import check_image_quality
from ..checkpoint_detect import get_model_type_for_checkpoint
from ..gpu_lock import gpu_lock, VideoGenerationActiveError, ClientDisconnectedWhileWaiting
from ..queue import get_queue
from ..abort import abort_controller
from ..safety import PromptPreprocessor

# Initialize global prompt filter
_prompt_filter = PromptPreprocessor()

logger = logging.getLogger(__name__)

generate_bp = Blueprint("generate", __name__)


def _prefetch_loras(loras):
    """Download LoRA files and warm the OS file cache before taking the GPU
    lock, so network/disk time doesn't block requests waiting for the GPU.
    The first load_lora_weights() call after a cold start can otherwise take
    tens of seconds inside the lock."""
    if not loras:
        return
    try:
        mgr = LoRAManager()
        for item in loras:
            url = item.get("url") if isinstance(item, dict) else None
            if not url:
                continue
            path = mgr.download_lora(url)
            # Warm OS page cache so the safetensors read under the lock is fast
            with open(path, "rb") as f:
                while f.read(8 * 1024 * 1024):
                    pass
    except Exception as e:
        logger.warning(f"LoRA prefetch failed (will retry under lock): {e}")


# TProxy workflow: fast FLUX.2-klein draft used as the img2img seed image.
# The endpoint URL must stay out of the repo: set the TPROXY_ENDPOINT env
# var, or define it in a gitignored routes/tproxy_local.py.
TPROXY_ENDPOINT = os.getenv("TPROXY_ENDPOINT", "")
if not TPROXY_ENDPOINT:
    try:
        from .tproxy_local import TPROXY_ENDPOINT as _local_endpoint
        TPROXY_ENDPOINT = _local_endpoint
    except ImportError:
        pass
TPROXY_STEPS = 4
TPROXY_GUIDANCE = 1.0
TPROXY_STRENGTH = 0.40


def _tproxy_draft(prompt, width=None, height=None, seed=None, steps=TPROXY_STEPS,
                   guidance=TPROXY_GUIDANCE, image_urls=None, timeout=300):
    """Generate a quick draft image via the TProxy edit API.

    Mirrors the multipart fields the service's web editor posts to its
    edit endpoint.
    image_urls: optional list of reference-image URLs appended as multipart
    "image" file parts (the editor endpoint expects all refs under "image").
    Returns a PIL image, or raises RuntimeError on any failure.
    """
    if not TPROXY_ENDPOINT:
        raise RuntimeError(
            "TProxy endpoint not configured: set the TPROXY_ENDPOINT env var "
            "or create routes/tproxy_local.py"
        )
    _origin = f"{urlparse(TPROXY_ENDPOINT).scheme}://{urlparse(TPROXY_ENDPOINT).netloc}"
    boundary = uuid.uuid4().hex

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    def file_field(name, filename, content):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + content + b"\r\n"

    body = b"".join([
        field("prompt", prompt),
        field("steps", str(steps)),
        field("guidance", str(guidance)),
        field("seed", str(seed if seed is not None else random.randint(0, 2**32 - 1))),
        field("num_images", "1"),
    ])
    for url in image_urls or []:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=timeout,
            ) as resp:
                content = resp.read()
        except Exception as e:
            raise RuntimeError(f"TProxy reference image download failed ({url}): {e}")
        filename = url.rsplit("/", 1)[-1].split("?")[0] or "reference.png"
        body += file_field("image", filename, content)
    if width:
        body += field("width", str(width))
    if height:
        body += field("height", str(height))
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        TPROXY_ENDPOINT,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            # The endpoint sits behind Cloudflare, which 403s the default
            # "Python-urllib" user-agent; send browser-like headers.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0 Safari/537.36",
            "Referer": _origin + "/",
            "Origin": _origin,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
    except Exception as e:
        raise RuntimeError(f"TProxy draft request failed: {e}")

    if not content_type.startswith("image/"):
        raise RuntimeError(
            f"TProxy draft returned {content_type}: {data[:300].decode(errors='replace')}"
        )
    return Image.open(BytesIO(data)).convert("RGB")


@generate_bp.route("/generate", methods=["POST"])
def generate():
    """
    Unified image generation endpoint with all features.
    
    JSON body:
        # Core parameters
        prompt (str): Required. Text prompt for generation.
        negative_prompt (str): Negative prompt (default: "").
        width (int): Image width (default: 1024).
        height (int): Image height (default: 1024).
        steps (int): Inference steps (default: 20).
        cfg_scale (float): Guidance scale (default: 7.0).
        seed (int): Random seed (default: random).
        
        # Model selection
        checkpoint (str): SDXL checkpoint filename.
        pipeline (str): Pipeline type: "sdxl" or "zimg" (default: "sdxl").
        
        # Scheduler options
        scheduler (str): Scheduler name (euler_a, dpm++_2m, etc.).
        use_karras (bool): Enable Karras sigmas (default: false).
        prediction_type (str): epsilon, v_prediction, sample (default: epsilon).
        
        # TIPO prompt enhancement
        use_tipo (bool): Enable TIPO tag conversion (default: false).
        tipo_max_tokens (int): Max TIPO tokens (default: 256).
        tipo_prepend_quality (bool): Prepend quality tags (default: true).
        
        # Hires fix (two-pass upscaling)
        use_hires_fix (bool): Enable hires fix (default: false).
        upscale_by (float): Upscale factor (default: 1.5).
        hires_denoising (float): Denoising strength (default: auto).
        
        # LoRA adapters
        loras (list): Array of {"url": "...", "strength": 1.0}.
        
        # Textual inversions
        embeddings (list): Array of {"url": "...", "token": "<name>", "strength": 1.0}.
        
        # Safety
        validate_output (bool): Check for artifacts (default: true).
        check_age (bool): Run age classifier on output (default: false).
            If potential minor detected, returns 451 error instead of image.
            Note: Banned words like "younger" are automatically filtered from prompts.
        
        # Output options
        monochrome (bool): Convert output to grayscale (black and white). Default: false.
        
        # TProxy workflow (two-stage: FLUX draft -> SDXL img2img)
        enableTProxy (bool): Draft via the TProxy API (4 steps,
            at the requested width/height), then refine with SDXL img2img
            using the rest of the passed settings. Default: false.
        tproxy_prompt (str): Optional text prepended to the regular prompt
            for the draft (default: none — the draft uses just "prompt").
        tproxy_strength (float): img2img strength for the TProxy refine pass
            (default 0.40; lower keeps more of the draft's detail).
        tproxy_steps (int): Steps for the FLUX draft (default 4, clamped 1-50).
        tproxy_guidance (float): Guidance for the FLUX draft (default 1.0,
            clamped 0-20).
        tproxy_image (str or list): Optional reference image URL (or list of
            URLs) forwarded to the FLUX draft as an edit input (multipart
            "image" fields). Default: none — pure text-to-image draft.
    
    Returns:
        PNG image on success, or JSON error on failure.
    """
    try:
        # Check if video generation is active
        if gpu_lock.is_video_active():
            return jsonify({
                "error": "Video generation in progress",
                "retry_after": 60
            }), 503
        
        # Pre-download LoRA files outside the GPU lock (network I/O)
        data = request.get_json() or {}
        _prefetch_loras(data.get("loras", []))
        
        # ====== TProxy workflow (draft before the GPU lock; network I/O) ======
        enable_tproxy = bool(data.get("enableTProxy") or data.get("enable_tproxy"))
        # Lower strength keeps more of the FLUX draft's detail (default 0.30).
        try:
            tproxy_strength = float(data.get("tproxy_strength", TPROXY_STRENGTH))
        except (TypeError, ValueError):
            tproxy_strength = TPROXY_STRENGTH
        tproxy_strength = min(max(tproxy_strength, 0.05), 0.95)
        try:
            tproxy_steps = int(data.get("tproxy_steps", TPROXY_STEPS))
        except (TypeError, ValueError):
            tproxy_steps = TPROXY_STEPS
        tproxy_steps = min(max(tproxy_steps, 1), 50)
        try:
            tproxy_guidance = float(data.get("tproxy_guidance", TPROXY_GUIDANCE))
        except (TypeError, ValueError):
            tproxy_guidance = TPROXY_GUIDANCE
        tproxy_guidance = min(max(tproxy_guidance, 0.0), 20.0)
        tproxy_draft = None
        if enable_tproxy:
            if data.get("pipeline") not in (None, "sdxl"):
                return jsonify({"error": "enableTProxy requires the sdxl pipeline"}), 400
            try:
                # The draft prompt is always the regular prompt; when a
                # tproxy_prompt is supplied it is prepended to it.
                base_prompt = (data.get("prompt") or "").strip()
                extra_prompt = (data.get("tproxy_prompt") or "").strip()
                draft_prompt = f"{extra_prompt} {base_prompt}".strip()
                # Reference image(s) for the draft: a single URL string or a
                # list of URLs, forwarded as multipart "image" fields.
                raw_image = data.get("tproxy_image") or data.get("tproxy_images")
                if isinstance(raw_image, str):
                    raw_image = [raw_image]
                image_urls = [u.strip() for u in (raw_image or []) if isinstance(u, str) and u.strip()]
                tproxy_draft = _tproxy_draft(
                    prompt=draft_prompt,
                    width=data.get("width"),
                    height=data.get("height"),
                    seed=data.get("seed"),
                    steps=tproxy_steps,
                    guidance=tproxy_guidance,
                    image_urls=image_urls,
                )
                logger.info(
                    f"TProxy draft ready: {tproxy_draft.width}x{tproxy_draft.height} "
                    f"({tproxy_steps} steps, guidance {tproxy_guidance}, "
                    f"{len(image_urls)} ref image(s) via {TPROXY_ENDPOINT})"
                )
            except RuntimeError as e:
                logger.error(f"TProxy draft failed: {e}")
                return jsonify({"error": str(e)}), 502
        
        with gpu_lock.acquire("generate"):
            
            # Validate required fields
            prompt = data.get("prompt")
            if not prompt:
                return jsonify({"error": "prompt is required"}), 400
            
            # Safety: Filter banned words from prompt
            has_banned, banned_found = _prompt_filter.contains_banned(prompt)
            if has_banned:
                logger.warning(f"Filtered banned words from prompt: {banned_found}")
                prompt = _prompt_filter.filter_prompt(prompt)
            
            # Core parameters
            negative_prompt = data.get("negative_prompt", "")
            # Also filter banned words from negative prompt
            if negative_prompt:
                negative_prompt = _prompt_filter.filter_prompt(negative_prompt)
            
            width = data.get("width")
            height = data.get("height")
            steps = data.get("steps")
            cfg_scale = data.get("cfg_scale") or data.get("guidance_scale")  # Accept both names
            seed = data.get("seed")
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            
            # Model selection
            checkpoint = data.get("checkpoint")
            pipeline_type = data.get("pipeline")  # Can be None for auto-detect
            
            # Auto-detect pipeline type from checkpoint if not specified
            if not pipeline_type and checkpoint:
                pipeline_type = get_model_type_for_checkpoint(checkpoint)
                logger.info(f"Auto-detected pipeline type: {pipeline_type} for checkpoint: {checkpoint}")
            elif not pipeline_type:
                pipeline_type = "sdxl"  # Default
            
            # Scheduler options
            scheduler = data.get("scheduler")
            use_karras = data.get("use_karras", False) or data.get("karras", False)
            prediction_type = data.get("prediction_type")  # None = use auto-detected
            
            # TIPO
            use_tipo = data.get("use_tipo", False)
            tipo_max_tokens = data.get("tipo_max_tokens", 256)
            tipo_prepend_quality = data.get("tipo_prepend_quality", True)
            
            # Hires fix
            use_hires_fix = data.get("use_hires_fix", False)
            upscale_by = data.get("upscale_by", 1.5)
            hires_denoising = data.get("hires_denoising")
            
            # Scale factor for first pass resolution scaling (0.25-2.0)
            # Accepts "scale" or legacy "down" parameter
            scale = data.get("scale", data.get("down", 1.0))
            if scale is not None:
                scale = float(scale)
                if scale < 0.25 or scale > 2.0:
                    return jsonify({"error": f"scale must be between 0.25 and 2.0, got {scale}"}), 400
                # Apply scale factor to dimensions
                if width and scale != 1.0:
                    width = round(width * scale / 16) * 16
                if height and scale != 1.0:
                    height = round(height * scale / 16) * 16
                logger.debug(f"Applied scale={scale}: {width}x{height}")
            
            # LoRAs and embeddings
            loras = data.get("loras", [])
            embeddings = data.get("embeddings", [])
            
            # Safety
            validate_output = data.get("validate_output", True)
            check_age = data.get("check_age", False)  # Age classifier check
            
            # Monochrome (black and white output)
            monochrome = data.get("monochrome", False)
            
            # ====== TIPO Enhancement ======
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
            
            # ====== Get Pipeline ======
            if pipeline_type == "zimg":
                # Z-Image can load from local checkpoints
                pipeline = get_pipeline("zimg", checkpoint=checkpoint)
            else:
                pipeline = get_pipeline("sdxl", checkpoint=checkpoint)
            
            # Ensure loaded
            if not pipeline.is_loaded:
                pipeline.load()
            
            # ====== Load LoRAs ======
            if loras and pipeline_type == "sdxl":
                try:
                    lora_manager = LoRAManager()
                    lora_manager.load_loras(pipeline._pipe, loras)
                except Exception as e:
                    logger.warning(f"LoRA loading failed: {e}")
            
            # ====== Load Embeddings ======
            if embeddings and pipeline_type == "sdxl":
                try:
                    emb_manager = EmbeddingManager()
                    emb_manager.load_embeddings(pipeline._pipe, embeddings)
                except Exception as e:
                    logger.warning(f"Embedding loading failed: {e}")
            
            # ====== Build generate kwargs ======
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
                # Only pass prediction_type if explicitly set (None = use auto-detected)
                if prediction_type:
                    generate_kwargs["prediction_type"] = prediction_type
                # Hires fix is now handled internally by the pipeline (uses prompt embeddings)
                if use_hires_fix:
                    generate_kwargs["use_hires_fix"] = True
                    generate_kwargs["upscale_by"] = upscale_by
                    if hires_denoising is not None:
                        generate_kwargs["hires_denoising"] = hires_denoising
                # Monochrome output
                if monochrome:
                    generate_kwargs["monochrome"] = True
            
            # ====== Generate with abort tracking ======
            abort_controller.start_generation()
            try:
                if tproxy_draft is not None:
                    # TProxy workflow: refine the FLUX draft; lower strength
                    # preserves more of the draft's detail. The rest of the
                    # passed settings ride along in generate_kwargs.
                    image = pipeline.img2img(
                        image=tproxy_draft,
                        strength=tproxy_strength,
                        **generate_kwargs,
                    )
                elif validate_output:
                    image = safe_generate(pipeline, generate_kwargs)
                else:
                    image = pipeline.generate(**generate_kwargs)
            finally:
                abort_controller.end_generation()
            
            # ====== Age Classifier Check ======
            if check_age:
                try:
                    from ..classifiers import get_classifier
                    age_classifier = get_classifier("age")
                    is_minor, confidence = age_classifier.check_is_minor(image)
                    if is_minor:
                        logger.warning(f"Age classifier flagged potential minor (confidence: {confidence:.2f})")
                        return jsonify({
                            "error": "Image flagged by age classifier",
                            "details": f"Potential minor detected with {confidence:.0%} confidence"
                        }), 451  # 451 Unavailable For Legal Reasons
                except Exception as e:
                    logger.warning(f"Age classifier failed: {e}")
            
            # ====== Return PNG Image ======
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            return Response(buffer.getvalue(), mimetype="image/png")
        
    except ClientDisconnectedWhileWaiting:
        logger.info("Client disconnected while waiting for GPU")
        return jsonify({"error": "Client disconnected"}), 499
    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"Generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "reason": str(e)}), 499
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        # Client disconnected - signal abort and return
        abort_controller.abort()
        logger.warning(f"Client disconnected during generation: {e}")
        return jsonify({"error": "Client disconnected"}), 499
    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": str(e)}), 500


@generate_bp.route("/generate/status", methods=["GET"])
def generate_status():
    """Get status of generation features."""
    tipo = get_tipo_engine()
    return jsonify({
        "tipo_loaded": tipo.is_loaded,
        "available_features": [
            "tipo", "hires_fix", "loras", "embeddings",
            "schedulers", "karras", "prediction_type", "tproxy"
        ]
    })


# ==================== Queue API Endpoints ====================

@generate_bp.route("/sdxl/queue", methods=["POST"])
def queue_generate():
    """
    Submit a generation request to the queue (async).
    
    Returns immediately with job ID and position.
    Use GET /job/<job_id> to check status and retrieve results.
    
    Returns:
        202 Accepted: {"job_id": "...", "status": "queued", "position": N}
        400 Bad Request: {"error": "..."} if validation fails
        503 Service Unavailable: {"error": "..."} if queue is full
    """
    try:
        # Check if video generation is active
        if gpu_lock.is_video_active():
            return jsonify({
                "error": "Video generation in progress - queue submissions blocked",
                "retry_after": 60
            }), 503
        
        data = request.get_json() or {}
        
        # Validate required fields
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        # Validate optional numeric parameters
        width = data.get("width")
        if width is not None and (not isinstance(width, int) or width < 64 or width > 4096):
            return jsonify({"error": "width must be an integer between 64 and 4096"}), 400
        
        height = data.get("height")
        if height is not None and (not isinstance(height, int) or height < 64 or height > 4096):
            return jsonify({"error": "height must be an integer between 64 and 4096"}), 400
        
        steps = data.get("steps")
        if steps is not None and (not isinstance(steps, int) or steps < 1 or steps > 150):
            return jsonify({"error": "steps must be an integer between 1 and 150"}), 400
        
        cfg_scale = data.get("cfg_scale") or data.get("guidance_scale")
        if cfg_scale is not None and (not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 30):
            return jsonify({"error": "cfg_scale must be a number between 0 and 30"}), 400
        
        # Submit to queue
        queue = get_queue()
        job = queue.submit(data)
        
        logger.info(f"Job {job.id[:8]} queued via /sdxl/queue")
        
        return jsonify({
            "job_id": job.id,
            "status": "queued",
            "position": job.position
        }), 202
        
    except RuntimeError as e:
        # Queue is full
        logger.warning(f"Queue submission failed: {e}")
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Queue submission error")
        return jsonify({"error": str(e)}), 500


@generate_bp.route("/job/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Get status of a queued/processing/completed job.
    
    Args:
        job_id: UUID of the job
        
    Returns:
        200 OK: Job status dict (status, result if done, error if failed)
        404 Not Found: {"error": "Job not found"} if job doesn't exist
    """
    queue = get_queue()
    job = queue.get_job(job_id)
    
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    
    response = job.to_dict()
    logger.debug(f"Job {job_id[:8]} status check: {job.status}")
    
    return jsonify(response), 200


@generate_bp.route("/queue/status", methods=["GET"])
def queue_status():
    """
    Get overall queue status.
    
    Returns:
        200 OK: {
            "queue_depth": N,
            "processing_count": N,
            "max_size": N,
            "queued": [...],
            "processing": [...],
            "recent_completed": [...]
        }
    """
    queue = get_queue()
    status = queue.get_queue_status()
    
    logger.debug(f"Queue status: depth={status['queue_depth']}, processing={status['processing_count']}")
    
    return jsonify(status), 200


@generate_bp.route("/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    """
    Cancel a job (queued or processing).
    
    Args:
        job_id: UUID of the job to cancel
        
    Returns:
        200 OK: {"status": "cancelled"} if successfully cancelled
        404 Not Found: {"error": "Job not found"} if job doesn't exist
        400 Bad Request: {"error": "..."} if job cannot be cancelled (already done, failed, etc.)
    """
    queue = get_queue()
    job = queue.get_job(job_id)
    
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    
    if job.status in ("done", "failed", "cancelled"):
        return jsonify({
            "error": f"Cannot cancel job in status '{job.status}'"
        }), 400
    
    # Attempt cancellation
    success = queue.cancel_job(job_id, abort_controller=abort_controller)
    
    if success:
        logger.info(f"Job {job_id[:8]} cancelled via API")
        return jsonify({"status": "cancelled"}), 200
    else:
        return jsonify({
            "error": f"Cannot cancel job in status '{job.status}'"
        }), 400
