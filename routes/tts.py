"""TTS routes for text-to-speech generation."""

import logging
import os
import tempfile
import time

import requests
from flask import Blueprint, request, jsonify, Response

from ..pipelines import get_pipeline
from ..abort import abort_controller, GenerationAbortedError
from ..gpu_lock import gpu_lock, VideoGenerationActiveError

logger = logging.getLogger(__name__)

tts_bp = Blueprint("tts", __name__)


@tts_bp.route("/generate", methods=["POST"])
def tts_generate():
    """
    Generate speech from text.
    
    JSON body:
        text (str): Required. Text to synthesize.
        voice_url (str): Optional. URL to voice reference audio for cloning.
        exaggeration (float): Optional. Emotion intensity (0-1, default 0.5).
        cfg_weight (float): Optional. CFG strength (0-1, default 0.5).
        turbo (bool): Optional. Use turbo model for ~80% faster inference (default False).
    
    Returns:
        WAV audio bytes with metadata headers.
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
        text = data.get("text")
        if not text:
            return jsonify({"error": "text is required"}), 400
        
        # Optional parameters
        voice_url = data.get("voice_url") or data.get("voice_audio_url")
        exaggeration = data.get("exaggeration", 0.5)
        cfg_weight = data.get("cfg_weight", 0.5)
        turbo = data.get("turbo", False)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.85)
        
        # Get TTS pipeline
        pipeline = get_pipeline("tts")
        
        # Ensure loaded
        if not pipeline.is_loaded:
            pipeline.load()
        
        # Handle voice URL - download to temp file
        audio_prompt_path = None
        temp_file = None
        
        try:
            if voice_url:
                logger.info(f"Downloading voice reference from: {voice_url}")
                response = requests.get(voice_url, timeout=30)
                response.raise_for_status()
                
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.write(response.content)
                temp_file.close()
                audio_prompt_path = temp_file.name
                logger.debug(f"Voice reference saved to: {audio_prompt_path}")
            
            # Generate audio with abort tracking
            start_time = time.time()
            
            abort_controller.start_generation()
            try:
                wav_bytes = pipeline.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    turbo=turbo,
                )
            finally:
                abort_controller.end_generation()
            
            gen_time = time.time() - start_time
            
            # Calculate audio duration (WAV header: sample rate at bytes 24-27, data size from file)
            # Rough estimate: ~24kHz sample rate, 16-bit mono
            audio_seconds = len(wav_bytes) / (24000 * 2)  # Approximate
            
            logger.info(f"TTS generated: {len(text)} chars -> {len(wav_bytes)} bytes in {gen_time:.2f}s")
            
            # Return WAV with metadata headers
            response = Response(wav_bytes, mimetype="audio/wav")
            response.headers["X-TTS-Model"] = "chatterbox"
            response.headers["X-TTS-Gen-Seconds"] = f"{gen_time:.3f}"
            response.headers["X-TTS-Audio-Seconds"] = f"{audio_seconds:.2f}"
            
            return response
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
    
    except (InterruptedError, GenerationAbortedError) as e:
        logger.info(f"TTS generation aborted: {e}")
        return jsonify({"error": "Generation aborted", "aborted": True}), 499
        
    except Exception as e:
        logger.exception("TTS generation failed")
        return jsonify({"error": str(e)}), 500


@tts_bp.route("/status", methods=["GET"])
def tts_status():
    """Get TTS pipeline status."""
    try:
        from ..pipelines import get_loaded_pipelines
        
        loaded_pipelines = get_loaded_pipelines()
        
        # Check if TTS is loaded
        tts_loaded = "tts" in loaded_pipelines
        tts_device = "cpu"
        
        if tts_loaded:
            pipeline = loaded_pipelines["tts"]
            tts_device = pipeline._device if hasattr(pipeline, '_device') else "cuda"
        
        return jsonify({
            "loaded": tts_loaded,
            "model": "chatterbox",
            "device": tts_device
        })
    except Exception as e:
        logger.exception("TTS status check failed")
        return jsonify({
            "loaded": False,
            "model": "chatterbox",
            "device": "unknown",
            "error": str(e)
        })
