"""Chatterbox TTS Pipeline implementation for text-to-speech generation."""

import gc
import io
import logging
import time
from typing import Optional

import torch

from .base import BasePipeline, pipeline_module

logger = logging.getLogger(__name__)


@pipeline_module(
    name="tts",
    display_name="Chatterbox TTS",
    output_type="audio",
    # Coexists with SDXL on 24GB (SDXL ~8GB + Chatterbox ~4.5GB); only actual
    # generation activations matter, and requests are short. ZImg/LTX still
    # conflict (LTX needs the full GPU).
    conflicts_with=["zimg", "ltx"],
    checkpoint_patterns=[],
    vram_estimate_gb=4.5,
    supports_checkpoints=False,
)
class TTSPipeline(BasePipeline):
    """Chatterbox TTS pipeline with VRAM management."""

    def __init__(self):
        self._model = None
        self._turbo_model = None
        self._device = "cpu"
        self._sample_rate: Optional[int] = None
        self._compiled: bool = False

    @property
    def is_loaded(self) -> bool:
        if self._model is None:
            return False
        return self._device == "cuda"

    @property
    def is_parked(self) -> bool:
        return self._model is not None and self._device == "cpu"

    def load(self) -> None:
        """Load Chatterbox model to CUDA."""
        if self.is_loaded:
            logger.debug("TTS already loaded, skipping")
            return

        start_time = time.time()
        logger.info("Loading Chatterbox TTS model")

        # Lazy import to avoid loading heavy dependencies until needed
        from chatterbox.tts import ChatterboxTTS

        # Clear memory before loading
        gc.collect()
        torch.cuda.empty_cache()

        # Load standard model
        self._model = ChatterboxTTS.from_pretrained(device="cuda")
        self._device = "cuda"
        self._sample_rate = self._model.sr

        # Try to load turbo model (may require HF auth)
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._turbo_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
            logger.info("Turbo model loaded successfully")
        except Exception as e:
            logger.warning(f"Turbo model unavailable (will use standard model): {e}")
            self._turbo_model = None

        load_time = time.time() - start_time
        logger.info(f"Chatterbox TTS loaded in {load_time:.2f}s (sample_rate={self._sample_rate})")

        # Apply torch.compile for faster inference
        if torch.cuda.is_available():
            try:
                # Compile the main inference components
                if hasattr(self._model, 't3') and self._model.t3 is not None:
                    self._model.t3 = torch.compile(self._model.t3, mode="reduce-overhead")
                if hasattr(self._model, 's2a') and self._model.s2a is not None:
                    self._model.s2a = torch.compile(self._model.s2a, mode="reduce-overhead")
                self._compiled = True
                logger.info("Applied torch.compile() to TTS components")
            except Exception as e:
                logger.warning(f"torch.compile() failed (will use eager mode): {e}")

        # Run warmup to trigger compilation
        self.warmup()

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._model is None and self._turbo_model is None:
            return

        logger.info("Unloading Chatterbox TTS models")

        # Delete both models
        del self._model
        del self._turbo_model
        self._model = None
        self._turbo_model = None
        self._device = "cpu"
        self._sample_rate = None

        # Aggressive cleanup
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("Chatterbox TTS unloaded")

    def _move_model_to_device(self, model, device: str) -> None:
        """Move a single model's components to the specified device."""
        if model is None:
            return
        if hasattr(model, 't3') and model.t3 is not None:
            model.t3.to(device)
        if hasattr(model, 've') and model.ve is not None:
            model.ve.to(device)
        if hasattr(model, 's2a') and model.s2a is not None:
            model.s2a.to(device)

    def to_cpu(self) -> None:
        """Move model to CPU RAM (parking)."""
        if (self._model is None and self._turbo_model is None) or self._device == "cpu":
            return

        logger.info("Parking TTS to CPU")
        
        # Move both models to CPU
        self._move_model_to_device(self._model, "cpu")
        self._move_model_to_device(self._turbo_model, "cpu")
        
        self._device = "cpu"

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("TTS parked to CPU")

    def to_gpu(self) -> None:
        """Move model from CPU back to GPU."""
        if (self._model is None and self._turbo_model is None) or self._device == "cuda":
            return

        logger.info("Restoring TTS to GPU")
        
        # Move both models to CUDA
        self._move_model_to_device(self._model, "cuda")
        self._move_model_to_device(self._turbo_model, "cuda")
        
        self._device = "cuda"
        logger.info("TTS restored to GPU")

    def warmup(self) -> None:
        """Run warmup generation to trigger torch.compile optimization."""
        if not self.is_loaded:
            return
        logger.info("Warming up TTS model...")
        try:
            with torch.inference_mode():
                _ = self._model.generate("Warmup test.")
            logger.info("TTS warmup complete")
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.7,
        top_p: float = 0.85,
        turbo: bool = False,
        **kwargs,
    ) -> bytes:
        """Generate speech from text.
        
        Args:
            text: Text to synthesize
            audio_prompt_path: Optional path to voice reference WAV for cloning
            exaggeration: Speech expressiveness (0-1, default 0.5)
            cfg_weight: Classifier-free guidance weight (0-1, default 0.5)
            temperature: Sampling temperature (default 0.7, lower = faster)
            top_p: Top-p sampling threshold (default 0.85, lower = faster)
            turbo: Use turbo model for ~80% faster inference (no emotion control)
            **kwargs: Additional arguments
            
        Returns:
            WAV audio bytes
        """
        self.touch()

        if not self.is_loaded:
            raise RuntimeError("TTS pipeline not loaded. Call load() first.")

        # Clamp parameters to valid range
        exaggeration = max(0.0, min(1.0, exaggeration))
        cfg_weight = max(0.0, min(1.0, cfg_weight))

        logger.debug(
            f"Generating TTS: text_len={len(text)}, voice_clone={audio_prompt_path is not None}, "
            f"turbo={turbo}, exaggeration={exaggeration}, cfg_weight={cfg_weight}, temperature={temperature}, top_p={top_p}"
        )

        # Pre-generation cleanup
        torch.cuda.empty_cache()

        # Generate with inference mode
        with torch.inference_mode():
            if turbo and self._turbo_model is not None:
                # Turbo mode: ~80% faster, no emotion control
                if audio_prompt_path:
                    wav = self._turbo_model.generate(text, audio_prompt_path=audio_prompt_path)
                else:
                    wav = self._turbo_model.generate(text)
            else:
                # Standard mode with full control
                if audio_prompt_path:
                    wav = self._model.generate(
                        text,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    wav = self._model.generate(
                        text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        top_p=top_p,
                    )

        # Convert tensor to WAV bytes
        wav_bytes = self._tensor_to_wav_bytes(wav)

        # Post-generation cleanup
        torch.cuda.empty_cache()

        logger.debug(f"Generated {len(wav_bytes)} bytes of audio")
        return wav_bytes

    def _tensor_to_wav_bytes(self, wav_tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes.
        
        Args:
            wav_tensor: Audio tensor from Chatterbox
            
        Returns:
            WAV file bytes
        """
        # Ensure tensor is on CPU and proper shape
        wav = wav_tensor.cpu()
        
        # Handle different tensor shapes
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension
        elif wav.dim() > 2:
            wav = wav.squeeze()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

        # Try torchaudio first, fallback to scipy
        try:
            import torchaudio
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, self._sample_rate, format="wav")
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            pass

        # Fallback to scipy
        try:
            import scipy.io.wavfile as wavfile
            import numpy as np
            
            # Convert to numpy int16
            wav_np = wav.squeeze().numpy()
            wav_np = np.clip(wav_np, -1.0, 1.0)
            wav_int16 = (wav_np * 32767).astype(np.int16)
            
            buffer = io.BytesIO()
            wavfile.write(buffer, self._sample_rate, wav_int16)
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            raise RuntimeError("Neither torchaudio nor scipy available for WAV encoding")

    def estimate_vram(self) -> int:
        """Estimate VRAM needed (~3GB standard, ~4.5GB with turbo)."""
        if self._turbo_model is not None:
            return int(4.5 * 1024 ** 3)
        return 3 * 1024 ** 3
