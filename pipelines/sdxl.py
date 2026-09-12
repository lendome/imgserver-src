"""SDXL Pipeline implementation using diffusers with proper VRAM optimization."""

import gc
import logging
import os
import sys
import time
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    DDPMScheduler,
    UniPCMultistepScheduler,
    PNDMScheduler,
    DEISMultistepScheduler,
    LCMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0

from .base import BasePipeline, pipeline_module
from ..config import get_config
from ..vram.manager import vram_manager
from ..vram.ram_manager import ram_manager
from ..checkpoint_detect import detect_checkpoint_config
from ..prompts import encode_long_prompt, is_long_prompt, tokenize_and_check
from ..lora import LoRAManager
from ..abort import abort_controller, GenerationAbortedError

logger = logging.getLogger(__name__)


# SageAttention: INT8-quantized QK attention (FP16 V/accumulation) on Triton.
# ~1% deviation from SDPA, meaningfully faster on Ampere. Pure-Triton build
# (sageattention 1.x), so it needs triton-windows on this machine.
try:
    from sageattention import sageattn as _sageattn
    SAGEATTN_AVAILABLE = True
except Exception as _sage_err:
    _sageattn = None
    SAGEATTN_AVAILABLE = False
    logger.info(f"SageAttention unavailable ({_sage_err}), will use SDPA/xformers")

SAGEATTN_SUPPORTED_HEAD_DIMS = (64, 96, 128)


class SageAttnProcessor:
    """Attention processor using SageAttention, with SDPA fallback.

    Mirrors diffusers' AttnProcessor2_0 contract but computes attention via
    sageattn (per-block INT8 QK quantization). Falls back to SDPA when the
    layer is unsupported (head dim not 64/96/128, masked attention, non-CUDA
    tensors) - SDXL's UNet only hits supported configs.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if (
            SAGEATTN_AVAILABLE
            and query.is_cuda
            and head_dim in SAGEATTN_SUPPORTED_HEAD_DIMS
            and attention_mask is None
        ):
            # sageattn handles the softmax scale internally (1/sqrt(head_dim))
            hidden_states = _sageattn(query, key, value, tensor_layout="HND", is_causal=False)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Set CUDA memory allocation config for better memory management
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8"
)

# Enable TF32 for faster computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Scheduler class mapping
SCHEDULER_CLASSES = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpm++_2m_sde": DPMSolverMultistepScheduler,
    "dpmpp_2m_sde": DPMSolverMultistepScheduler,
    "dpm++_sde": DPMSolverSinglestepScheduler,
    "dpmpp_sde": DPMSolverSinglestepScheduler,
    "dpm_2": KDPM2DiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm_2_a": KDPM2AncestralDiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "unipc": UniPCMultistepScheduler,
    "pndm": PNDMScheduler,
    "deis": DEISMultistepScheduler,
    "lcm": LCMScheduler,
}

KARRAS_COMPATIBLE = {
    "euler", "euler_a", "euler_ancestral",
    "dpm++_2m", "dpmpp_2m", "dpm++_2m_sde", "dpmpp_2m_sde",
    "dpm++_sde", "dpmpp_sde", "dpm_2", "dpm2", "dpm_2_a", "dpm2_a",
    "heun", "lms", "deis",
}

SDE_SCHEDULERS = {"dpm++_2m_sde", "dpmpp_2m_sde", "dpm++_sde", "dpmpp_sde"}


class VAEStateGuard:
    """Context manager to save/restore VAE state, preventing dtype/config poisoning.
    
    This ensures that even if an exception occurs during VAE decode, the VAE
    state will be restored to its original dtype and device configuration.
    """
    
    def __init__(self, vae):
        self.vae = vae
        self.original_dtype = None
        self.original_device = None
        self.original_force_upcast = None
        
    def __enter__(self):
        try:
            param = next(self.vae.parameters())
            self.original_dtype = param.dtype
            self.original_device = param.device
        except StopIteration:
            pass
        # Cache config state
        self.original_force_upcast = getattr(self.vae.config, 'force_upcast', None)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always restore state, even on exception
        try:
            if self.original_dtype is not None:
                current_dtype = next(self.vae.parameters()).dtype
                if current_dtype != self.original_dtype:
                    self.vae.to(dtype=self.original_dtype)
                    logger.debug(f"VAE dtype restored: {current_dtype} -> {self.original_dtype}")
        except Exception as e:
            logger.warning(f"Failed to restore VAE dtype: {e}")
        return False  # Don't suppress exceptions


def _upscale_latents(
    latents: torch.Tensor, 
    scale_factor: float, 
    mode: str = 'bicubic',
    antialiasing_noise: float = 0.02
) -> torch.Tensor:
    """Upscale latents directly in latent space with detail preservation.
    
    This is MUCH faster than decode->upscale->encode because it skips 
    2 expensive VAE operations (~1.5s savings).
    
    Uses bicubic interpolation (sharper than bilinear) plus a small amount
    of high-frequency noise to restore detail lost during upscaling.
    
    Args:
        latents: Input latents [B, 4, H, W]
        scale_factor: Upscale factor (e.g., 1.5 for 1.5x)
        mode: Interpolation mode (bicubic recommended for sharpness)
        antialiasing_noise: Small noise to restore high-freq detail (0.02 = 2%)
        
    Returns:
        Upscaled latents [B, 4, H*scale, W*scale]
    """
    upscaled = torch.nn.functional.interpolate(
        latents,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=False if mode in ('bilinear', 'bicubic') else None
    )
    
    # Add small amount of noise to restore high-frequency detail
    # This compensates for the smoothing effect of interpolation
    if antialiasing_noise > 0:
        noise = torch.randn_like(upscaled) * antialiasing_noise
        upscaled = upscaled + noise
    
    return upscaled


# Pinned memory buffer cache for efficient GPU->CPU transfer
_pinned_buffer_cache: Dict[tuple, torch.Tensor] = {}
MAX_PINNED_BUFFERS = 4  # Limit cache size to avoid excessive RAM usage


def _get_pinned_buffer(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Get or create a pinned memory buffer for efficient GPU->CPU transfer."""
    key = (shape, dtype)
    if key not in _pinned_buffer_cache:
        # Evict oldest if at capacity
        if len(_pinned_buffer_cache) >= MAX_PINNED_BUFFERS:
            oldest_key = next(iter(_pinned_buffer_cache))
            del _pinned_buffer_cache[oldest_key]
        _pinned_buffer_cache[key] = torch.empty(shape, dtype=dtype, pin_memory=True)
    return _pinned_buffer_cache[key]


def _clear_pinned_buffer_cache():
    """Clear pinned memory cache to free system RAM."""
    global _pinned_buffer_cache
    _pinned_buffer_cache.clear()


def _decode_latents_to_image(
    vae, 
    latents: torch.Tensor, 
    return_pil: bool = True
) -> Union[Image.Image, torch.Tensor]:
    """Decode latents to PIL image with maximum throughput.
    
    Respects VAE force_upcast config - some VAEs (e.g. novaRealityXL) require 
    FP32 decode to avoid NaN. The VAE config has force_upcast=True for these models.
    
    Performance optimizations (vs previous version):
    1. force_upcast only upcasts decoder submodules instead of entire VAE
       (~50% less weight copy per decode call on force_upcast models)
    2. Disables VAE tiling/slicing for standard resolutions (<=1536px) since 
       SDXL only uses ~7GB of 24GB VRAM - tiling adds overhead for no benefit
    3. Enables tiling only for large resolutions where it's actually needed
    4. Single synchronization point at the end instead of redundant syncs
    5. Fused GPU post-processing: normalize + uint8 in one pass
    6. Multiplication instead of division for latent scaling
    
    Args:
        vae: The VAE decoder model
        latents: Latent tensor to decode
        return_pil: If True, return PIL Image. If False, return uint8 tensor (NHWC format)
    
    IMPORTANT: Uses VAEStateGuard to ensure VAE state is always restored,
    even if an exception occurs during decode.
    """
    decode_start = time.time()
    
    # Use VAEStateGuard to ensure dtype/device restoration on any exit path
    with VAEStateGuard(vae):
        vae_device = next(vae.parameters()).device
        original_dtype = next(vae.parameters()).dtype
        
        # Validate VAE state before decode
        if vae_device.type != "cuda":
            logger.warning(f"VAE on unexpected device: {vae_device}, expected cuda")
        
        # Check if VAE needs FP32 decode (force_upcast config)
        force_upcast = getattr(vae.config, 'force_upcast', False)
        
        # Get VAE scaling factor (SDXL default is 0.13025)
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
        
        # Scale latents (use multiply instead of divide - faster)
        inv_scaling = 1.0 / scaling_factor
        latents = latents * inv_scaling
        
        # Determine if we need tiling based on latent resolution
        # Latent space is 1/8 of pixel space, so 128 latent = 1024px, 192 = 1536px
        latent_h, latent_w = latents.shape[-2], latents.shape[-1]
        pixel_h, pixel_w = latent_h * 8, latent_w * 8
        needs_tiling = (pixel_h > 1536 or pixel_w > 1536)
        
        if needs_tiling:
            # Large image: enable tiling to prevent OOM
            vae.enable_tiling()
            vae.enable_slicing()
            logger.debug(f"VAE tiling enabled for {pixel_w}x{pixel_h}")
        else:
            # Standard resolution: disable tiling/slicing for speed
            # SDXL uses ~7GB of 24GB VRAM - we have plenty of headroom
            # Tiling adds kernel launch overhead and reduces GPU parallelism
            vae.disable_tiling()
            vae.disable_slicing()
        
        with torch.no_grad():
            if force_upcast:
                # Use autocast to run VAE decode in FP32 WITHOUT copying weights
                # This is ~10x faster than vae.to(float32) + vae.to(float16)
                # autocast handles the upcast at the operation level using Tensor Cores
                latents = latents.to(device=vae_device, dtype=torch.float32)
                with torch.cuda.amp.autocast(enabled=False):
                    # Temporarily upcast only the VAE decoder weights that matter
                    # Use the VAE's own upcast_vae method if available, else manual
                    upcast_dtype = vae.dtype
                    vae.post_quant_conv.to(dtype=torch.float32)
                    vae.decoder.conv_in.to(dtype=torch.float32)
                    vae.decoder.mid_block.to(dtype=torch.float32)
                    # Decode up blocks need fp32 too for numerical stability
                    for up_block in vae.decoder.up_blocks:
                        up_block.to(dtype=torch.float32)
                    vae.decoder.conv_norm_out.to(dtype=torch.float32)
                    vae.decoder.conv_out.to(dtype=torch.float32)
                    
                    decoded = vae.decode(latents, return_dict=False)[0]
                    
                    # Restore decoder to original dtype (only decoder, not full VAE)
                    vae.post_quant_conv.to(dtype=upcast_dtype)
                    vae.decoder.conv_in.to(dtype=upcast_dtype)
                    vae.decoder.mid_block.to(dtype=upcast_dtype)
                    for up_block in vae.decoder.up_blocks:
                        up_block.to(dtype=upcast_dtype)
                    vae.decoder.conv_norm_out.to(dtype=upcast_dtype)
                    vae.decoder.conv_out.to(dtype=upcast_dtype)
            else:
                # Normal FP16 decode - fastest path
                latents = latents.to(device=vae_device, dtype=original_dtype)
                decoded = vae.decode(latents, return_dict=False)[0]
        
        # Check for NaN (fast - single reduction kernel)
        if decoded.isnan().any():
            logger.error("NaN detected in VAE output - this indicates VAE corruption")
        
        # Fused GPU post-processing (faster than separate ops)
        # Normalize [-1,1] to [0,255] and convert to uint8 on GPU in one pass
        # Math: (x / 2 + 0.5) * 255 = (x + 1) * 127.5
        decoded = ((decoded + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # Permute to NHWC format and ensure contiguous for efficient CPU transfer
        decoded = decoded.permute(0, 2, 3, 1).contiguous()
        
        if not return_pil:
            return decoded
        
        # Fast CPU transfer with pinned memory (async copy)
        pinned_buf = _get_pinned_buffer(decoded.shape, decoded.dtype)
        pinned_buf.copy_(decoded, non_blocking=True)
        torch.cuda.synchronize()  # Single sync point - wait for async copy
        
        # Numpy view of pinned memory, then copy out so the returned image owns
        # its memory. Zero-copy would alias the cached buffer, which the next
        # decode reuses and overwrites - corrupting any image still held by the
        # caller (e.g. during quality-check retries or concurrent requests).
        decoded_np = pinned_buf.numpy()[0].copy()
        
        decode_time = time.time() - decode_start
        logger.debug(f"VAE decode completed in {decode_time:.3f}s ({pixel_w}x{pixel_h}, "
                     f"force_upcast={force_upcast}, tiling={needs_tiling})")
        
        return Image.fromarray(decoded_np)
    # VAE state guaranteed restored here by VAEStateGuard



@pipeline_module(
    name="sdxl",
    display_name="SDXL Image Generation",
    output_type="image",
    # TTS (Chatterbox ~4.5GB) coexists with SDXL (~8GB) on 24GB - no unload.
    conflicts_with=["zimg", "ltx"],  # Must unload these before loading SDXL
    checkpoint_patterns=[r"sdxl", r"xl", r"pony", r"illustrious", r"animagine", r"noob", r"wai", r"juggernaut", r"dreamshaper"],
    vram_estimate_gb=8.0,
    supports_checkpoints=True
)
class SDXLPipeline(BasePipeline):
    """SDXL pipeline with optimized VRAM management."""

    # CPU fp16 state dicts of the SDXL-base text encoders, loaded lazily the
    # first time a checkpoint without baked-in text encoders needs them.
    _base_te_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    def __init__(self, checkpoint: Optional[str] = None):
        self._pipe: Optional[StableDiffusionXLPipeline] = None
        self._refiner_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
        self._checkpoint = checkpoint
        self._config = get_config()
        self._parked = False
        
        # Detect model configuration (pass full path for metadata reading)
        full_path = str(self._config.get_checkpoint_path(checkpoint)) if checkpoint else ""
        self._checkpoint_config = detect_checkpoint_config(full_path)
        self._prediction_type = self._checkpoint_config.prediction_type
        logger.info(f"SDXL pipeline initialized: checkpoint={checkpoint}, prediction_type={self._prediction_type}")

    def _create_abort_callback(self):
        """Create callback that checks for abort request each step."""
        def callback(pipeline, step_index, timestep, callback_kwargs):
            if abort_controller.should_abort():
                logger.warning(f"Generation aborted at step {step_index}")
                raise GenerationAbortedError(f"Aborted at step {step_index}")
            return callback_kwargs
        return callback

    @property
    def is_loaded(self) -> bool:
        if self._pipe is None:
            return False
        try:
            return self._pipe.device.type == "cuda"
        except Exception:
            return False

    @property
    def is_parked(self) -> bool:
        return self._parked and self._pipe is not None

    @property
    def checkpoint_path(self) -> str:
        return str(self._config.get_checkpoint_path(self._checkpoint))
    
    @property
    def prediction_type(self) -> str:
        return self._prediction_type

    def _apply_optimizations(self) -> None:
        """Apply all VRAM optimizations to the pipeline."""
        pipe = self._pipe
        
        # 1. Enable VAE slicing (will be toggled per-decode based on resolution)
        pipe.vae.enable_slicing()
        logger.debug("VAE slicing enabled (default - may be disabled for small images)")
        
        # 2. VAE tiling is managed dynamically per-decode based on resolution
        # Don't enable globally - it slows down standard resolutions
        # _decode_latents_to_image() enables tiling only when needed (>1536px)
        logger.debug("VAE tiling: managed per-decode (enabled only for >1536px)")
        
        # 3. Convert to channels_last memory format (faster on modern GPUs)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        logger.debug("Channels-last memory format enabled")
        
        # 4. Ensure fp16 dtype for UNet (VAE dtype managed by force_upcast config)
        pipe.unet.to(dtype=torch.float16)
        # Note: DO NOT force VAE to FP16 - some models need FP32 decode (force_upcast=True)
        
        # 5. Attention backend: SageAttention (INT8 QK) > xformers > SDPA
        sage_active = False
        if getattr(self._config, "sage_attention_enabled", False) and SAGEATTN_AVAILABLE:
            try:
                # NB: set_attn_processor pops entries from the dict as it
                # attaches them, so capture the count before the call.
                processor_map = {name: SageAttnProcessor() for name in pipe.unet.attn_processors}
                num_layers = len(processor_map)
                pipe.unet.set_attn_processor(processor_map)
                sage_active = True
                logger.info(
                    f"SageAttention enabled for UNet ({num_layers} attention layers; "
                    "SDPA fallback for unsupported head dims)"
                )
            except Exception as e:
                logger.warning(f"SageAttention setup failed, falling back to xformers/SDPA: {e}")

        if not sage_active:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled")
            except Exception:
                # Fallback to PyTorch 2.0 SDPA
                try:
                    pipe.unet.set_attn_processor(AttnProcessor2_0())
                    logger.info("SDPA attention enabled (xformers unavailable)")
                except Exception as e:
                    logger.warning(f"Could not enable optimized attention: {e}")

        # 6. Fuse QKV projections for speed.
        #    Skipped under SageAttention: fuse_qkv_projections replaces custom
        #    processors with FusedAttnProcessor2_0, which would drop the sage kernel.
        if not sage_active:
            try:
                pipe.fuse_qkv_projections()
                logger.debug("QKV projections fused")
            except Exception:
                pass

    def _configure_scheduler_for_prediction_type(self) -> None:
        """Configure scheduler at load time based on detected prediction_type.
        
        CRITICAL: This must be done at load time, not at generation time.
        The old server does this immediately after loading the checkpoint.
        Uses a FRESH copy of scheduler config to prevent contamination.
        """
        if self._pipe is None:
            return
        
        # Get FRESH copy of scheduler config (prevents reference to stale configs)
        fresh_scheduler_config = dict(self._pipe.scheduler.config)
        
        if self._prediction_type == "v_prediction":
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                fresh_scheduler_config,
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True,
                use_karras_sigmas=False
            )
            logger.info("Scheduler configured for v_prediction: trailing spacing, rescale_betas_zero_snr=True")
        else:  # epsilon
            self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                fresh_scheduler_config,
                prediction_type="epsilon",
                timestep_spacing="leading",
                rescale_betas_zero_snr=False,
                use_karras_sigmas=False
            )
            logger.info("Scheduler configured for epsilon: leading spacing")
        
        # Verify configuration
        actual_pred = getattr(self._pipe.scheduler.config, 'prediction_type', None)
        if actual_pred != self._prediction_type:
            logger.warning(f"Scheduler prediction_type mismatch: expected {self._prediction_type}, got {actual_pred}")
            # Force it
            self._pipe.scheduler.config.prediction_type = self._prediction_type

    def _should_compile_model(self) -> bool:
        """Check if torch.compile should be applied for speedup.

        Returns:
            True if compilation is enabled and the system supports it.
        """
        if not getattr(self._config, 'torch_compile_enabled', False):
            return False

        try:
            import torch
            if not hasattr(torch, 'compile'):
                logger.info("torch.compile not available (PyTorch < 2.0)")
                return False

            if not torch.cuda.is_available():
                return False

            # Inductor needs Triton for GPU codegen; on Windows that means
            # the triton-windows package matching the installed torch.
            if sys.platform == "win32":
                import importlib.util
                if importlib.util.find_spec("triton") is None:
                    logger.info("torch.compile skipped: triton not installed "
                                "(Windows requires triton-windows matching torch)")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Compilation check failed: {e}")
            return False

    def _compile_for_speed(self) -> None:
        """Compile the UNet with torch.compile (Inductor + Triton).

        The first generation after load pays a one-time compile cost
        (minutes, cold); Inductor's FX graph cache persists compiled kernels
        on disk across restarts, so subsequent server starts are faster.
        Steady-state UNet steps typically run 15-30% faster than eager.

        Note: PEFT LoRA loading (load_lora_weights) mutates module structure
        after compilation, which forces a one-time recompile per LoRA state -
        not an error. cache_size_limit is raised so varied resolutions and
        LoRA states don't exhaust the code cache and silently disable
        compilation mid-session.
        """
        if self._pipe is None:
            return

        try:
            import torch._dynamo
            import torch._inductor.config as inductor_config

            # Multiple LoRA states can still force occasional retraces; the
            # default limit of 8 would give up early and silently disable
            # compilation mid-session.
            torch._dynamo.config.cache_size_limit = 64
            # Persist generated kernels across process restarts.
            inductor_config.fx_graph_cache = True

            start = time.time()
            # dynamic=True: one symbolic-shape graph serves ALL resolutions.
            # With static shapes, every new width/height (e.g. 832x1216 after
            # 1024x1024) forced a full multi-minute dynamo+inductor recompile
            # on the first step - unacceptable for a hotswap server.
            self._pipe.unet = torch.compile(self._pipe.unet, dynamic=True)
            logger.info(f"UNet wrapped in torch.compile(dynamic=True) "
                        f"(setup {time.time() - start:.2f}s; first generation compiles, "
                        "may take a few minutes)")
        except Exception as e:
            logger.warning(f"torch.compile setup failed, running eager: {e}")

    def load(self) -> None:
        """Load SDXL model with optimizations."""
        if self.is_loaded:
            logger.debug("SDXL already loaded, skipping")
            return

        # Clear memory before loading
        vram_manager.cleanup()
        
        needed_ram = self.estimate_vram()
        if not ram_manager.can_park_model(needed_ram):
            logger.warning(f"Low RAM for SDXL load: need {needed_ram/1e9:.1f}GB")

        start_time = time.time()
        logger.info(f"Loading SDXL from {self.checkpoint_path}")
        
        # Load with fp16 and optimizations
        # CRITICAL: Explicitly specify config to prevent diffusers from
        # incorrectly inferring model architecture (e.g., using Lumina-Image instead of SDXL)
        # Some checkpoints don't include text encoders - we load them from base model
        from diffusers.loaders.single_file_utils import SingleFileComponentError
        
        try:
            self._pipe = StableDiffusionXLPipeline.from_single_file(
                self.checkpoint_path,
                config="stabilityai/stable-diffusion-xl-base-1.0",  # Force SDXL architecture
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,  # Faster loading, less RAM spike
            )
        except SingleFileComponentError as e:
            # Checkpoint missing text encoders - load them from base model
            logger.info(f"Checkpoint missing components, loading from base: {e}")
            from transformers import CLIPTextModel, CLIPTextModelWithProjection
            
            text_encoder = CLIPTextModel.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="text_encoder",
                torch_dtype=torch.float16,
            )
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="text_encoder_2",
                torch_dtype=torch.float16,
            )
            
            self._pipe = StableDiffusionXLPipeline.from_single_file(
                self.checkpoint_path,
                config="stabilityai/stable-diffusion-xl-base-1.0",
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        
        load_time = time.time() - start_time
        logger.info(f"Checkpoint loaded from disk in {load_time:.2f}s")
        
        # Move to CUDA
        cuda_start = time.time()
        self._pipe.to("cuda")
        self._parked = False
        cuda_time = time.time() - cuda_start
        logger.info(f"Moved to CUDA in {cuda_time:.2f}s")
        
        # Apply all optimizations
        self._apply_optimizations()
        
        # CRITICAL: Configure scheduler IMMEDIATELY after load with FRESH config
        # This is how the old server does it - scheduler must be set at load time
        # with correct prediction_type, not at generation time
        self._configure_scheduler_for_prediction_type()
        
        # Create refiner pipeline (shares ALL components, no extra VRAM)
        # This is needed for hires fix img2img pass
        self._refiner_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            text_encoder_2=self._pipe.text_encoder_2,
            tokenizer=self._pipe.tokenizer,
            tokenizer_2=self._pipe.tokenizer_2,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
        )
        logger.debug("Refiner (img2img) pipeline created with shared components")
        
        # Apply torch.compile optimization for faster inference
        if self._should_compile_model():
            self._compile_for_speed()
        
        # Final cleanup
        vram_manager.cleanup()
        
        total_time = time.time() - start_time
        logger.info(f"SDXL loaded with optimizations in {total_time:.2f}s")

    def unload(self) -> None:
        """Unload pipeline and free VRAM with component-by-component cleanup."""
        if self._pipe is None:
            return

        logger.info("Unloading SDXL pipeline")
        
        # CRITICAL: Clean LoRA state before unload to prevent weight poisoning
        try:
            if self._pipe is not None:
                LoRAManager().prepare_for_checkpoint_switch(self._pipe)
        except RuntimeError as e:
            logger.warning(f"LoRA cleanup failed during unload (will reload fresh): {e}")
        except Exception as e:
            logger.debug(f"Could not prepare LoRA state for unload: {e}")
        finally:
            # Always reset tracking even if cleanup failed
            LoRAManager().reset_loaded_adapters()
        
        # CRITICAL: Reset VAE state to prevent dtype/config poisoning across checkpoints
        try:
            self._reset_vae_state()
        except Exception as e:
            logger.debug(f"Could not reset VAE state during unload: {e}")
        
        # Delete refiner first (shares components with main pipe)
        if self._refiner_pipe is not None:
            del self._refiner_pipe
            self._refiner_pipe = None
        
        # Component-by-component cleanup (faster VRAM release than full .to("cpu"))
        self._component_cleanup()
        
        # Delete pipeline
        del self._pipe
        self._pipe = None
        self._parked = False
        
        # Aggressive cleanup
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        logger.info("SDXL unloaded")

    def _component_cleanup(self) -> None:
        """Delete individual components to free VRAM (no CPU move to avoid fp16 warnings)."""
        if self._pipe is None:
            return

        # Order matters: UNet is biggest, free it first
        components = ['unet', 'vae', 'text_encoder', 'text_encoder_2']

        for comp_name in components:
            comp = getattr(self._pipe, comp_name, None)
            if comp is not None:
                try:
                    # Don't move to CPU - just delete directly (avoids fp16 warning)
                    setattr(self._pipe, comp_name, None)
                    del comp
                    # Incremental cleanup between components for faster VRAM release
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Could not cleanup {comp_name}: {e}")

    # ------------------------------------------------------------------ #
    # Hot-swap: change checkpoint without losing torch.compile            #
    # ------------------------------------------------------------------ #

    @classmethod
    def _base_te_state_dicts(cls) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load (once) and cache SDXL-base text encoder weights on CPU.

        Needed when switching to a checkpoint that ships without baked-in
        text encoders while the currently loaded one has them.
        """
        if cls._base_te_cache is None:
            from transformers import CLIPTextModel, CLIPTextModelWithProjection

            logger.info("Loading SDXL-base text encoders for hot-swap cache...")
            te1 = CLIPTextModel.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="text_encoder", torch_dtype=torch.float16,
            )
            te2 = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="text_encoder_2", torch_dtype=torch.float16,
            )
            cls._base_te_cache = {
                "te1": {k: v.detach().clone() for k, v in te1.state_dict().items()},
                "te2": {k: v.detach().clone() for k, v in te2.state_dict().items()},
            }
            del te1, te2
        return cls._base_te_cache

    @staticmethod
    def _load_strict(module: "torch.nn.Module", state_dict: Dict[str, "torch.Tensor"], what: str) -> None:
        """load_state_dict with strict validation; benign buffers ignored.

        Raises RuntimeError on any real mismatch so the caller can fall back
        to a full reload.
        """
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        if missing or unexpected:
            raise RuntimeError(
                f"{what} state mismatch after conversion "
                f"(missing={len(missing)}, unexpected={len(unexpected)}; "
                f"first missing={missing[:3]}, first unexpected={unexpected[:3]})"
            )

    def switch_checkpoint(self, checkpoint: Optional[str]) -> None:
        """Hot-swap to another checkpoint WITHOUT reloading the pipeline.

        Converts the new single-file checkpoint's state dicts and copies them
        into the existing modules. Because parameter objects are reused,
        torch.compile's guards stay valid: no dynamo retrace, no inductor
        recompile, no VRAM churn. Typically ~10-20s vs 60-90s (warm cache)
        for a full reload + recompile.

        Raises on any mismatch (incompatible architecture, corrupt weights);
        the caller is expected to fall back to unload + full load.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded")

        name = checkpoint or self._config.default_checkpoint
        path = str(self._config.get_checkpoint_path(name))

        start = time.time()
        logger.info(f"Hot-swapping SDXL checkpoint -> {name}")

        # Strip any LoRA adapters first - they wrap modules and would break
        # strict state-dict validation (mirrors unload()'s cleanup order).
        try:
            LoRAManager().prepare_for_checkpoint_switch(self._pipe)
        except RuntimeError as e:
            raise RuntimeError(f"LoRA cleanup failed, cannot hot-swap safely: {e}")
        finally:
            LoRAManager().reset_loaded_adapters()

        from safetensors.torch import load_file
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
            convert_ldm_unet_checkpoint,
            convert_ldm_vae_checkpoint,
        )

        raw = load_file(path)

        if not any(k.startswith("model.diffusion_model.") for k in raw):
            raise RuntimeError(f"No UNet weights ('model.diffusion_model.*') in {name}")

        # --- UNet (target the real module under the torch.compile wrapper) ---
        unet_module = getattr(self._pipe.unet, "_orig_mod", self._pipe.unet)
        unet_sd = convert_ldm_unet_checkpoint(raw, dict(unet_module.config))
        self._load_strict(unet_module, unet_sd, "UNet")
        del unet_sd

        # --- VAE ---
        if any(k.startswith("first_stage_model.") for k in raw):
            vae_sd = convert_ldm_vae_checkpoint(raw, dict(self._pipe.vae.config))
            self._load_strict(self._pipe.vae, vae_sd, "VAE")
            del vae_sd

        # --- Text encoders (restore SDXL-base TEs when not baked in) ---
        p1 = "conditioner.embedders.0.transformer."
        p2 = "conditioner.embedders.1.transformer."
        te1_sd = {k[len(p1):]: v for k, v in raw.items() if k.startswith(p1)}
        te2_sd = {k[len(p2):]: v for k, v in raw.items() if k.startswith(p2)}
        if not te1_sd or not te2_sd:
            base = SDXLPipeline._base_te_state_dicts()
            if not te1_sd:
                te1_sd = base["te1"]
            if not te2_sd:
                te2_sd = base["te2"]
        self._load_strict(self._pipe.text_encoder, te1_sd, "text_encoder")
        self._load_strict(self._pipe.text_encoder_2, te2_sd, "text_encoder_2")
        del raw, te1_sd, te2_sd

        # --- Refresh checkpoint-dependent state ---
        self._checkpoint = name
        self._checkpoint_config = detect_checkpoint_config(path)
        self._prediction_type = self._checkpoint_config.prediction_type
        self._reset_vae_state()
        self._configure_scheduler_for_prediction_type()
        if self._refiner_pipe is not None:
            self._refiner_pipe.scheduler = self._pipe.scheduler

        gc.collect()
        logger.info(
            f"SDXL hot-swapped to {name} in {time.time() - start:.2f}s "
            f"(prediction_type={self._prediction_type}, torch.compile preserved)"
        )

    def _clear_scheduler_state(self) -> None:
        """Clear scheduler internal state to prevent cross-checkpoint contamination."""
        if self._pipe is None or self._pipe.scheduler is None:
            return
        
        scheduler = self._pipe.scheduler
        
        # Clear all cached state that could cause artifacts on checkpoint switch
        state_attrs = [
            'timesteps', 'sigmas', 'num_inference_steps',
            'betas', 'alphas', 'alphas_cumprod', 'one',
            '_step_index', '_begin_index', 'init_noise_sigma',
            'model_outputs', 'lower_order_nums', 'sample',
        ]
        
        for attr in state_attrs:
            if hasattr(scheduler, attr):
                try:
                    delattr(scheduler, attr)
                except (AttributeError, TypeError):
                    pass
        
        logger.debug("Scheduler state cleared")

    def to_cpu(self) -> None:
        """Park pipeline to CPU RAM."""
        if self._pipe is None or self._parked:
            return
        
        logger.info("Parking SDXL to CPU")
        self._pipe.to("cpu")
        # Refiner shares components, so it moves automatically
        self._parked = True
        
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SDXL parked to CPU")

    def to_gpu(self) -> None:
        """Restore pipeline from CPU to GPU."""
        if self._pipe is None or not self._parked:
            return
        
        logger.info("Restoring SDXL to GPU")
        self._pipe.to("cuda")
        # Refiner shares components, so it moves automatically
        self._parked = False
        logger.info("SDXL restored to GPU")

    def _create_abort_callback(self):
        """Create callback to check abort flag during generation.
        
        Returns a callback function that checks abort_controller.should_abort()
        and raises InterruptedError to stop generation.
        """
        def callback(pipeline, step_index, timestep, callback_kwargs):
            if abort_controller.should_abort():
                logger.warning(f"Generation aborted at step {step_index}")
                raise InterruptedError("Generation aborted by client")
            return callback_kwargs
        return callback

    def _ensure_vae_ready(self) -> None:
        """Ensure VAE is in correct state for decode.
        
        Sets eval mode. Does NOT force dtype or tiling/slicing - those are
        managed dynamically per-decode based on resolution and force_upcast.
        """
        if self._pipe is None or self._pipe.vae is None:
            return
        
        vae = self._pipe.vae
        
        # Ensure VAE is in eval mode
        vae.eval()

    def _reset_vae_state(self) -> None:
        """Reset VAE state to prevent dtype/config poisoning across checkpoints.
        
        Called during checkpoint switch to ensure clean VAE state.
        """
        if self._pipe is None or self._pipe.vae is None:
            return
        
        vae = self._pipe.vae
        
        # Reset VAE to eval mode
        vae.eval()
        
        # Clear any cached internal state
        if hasattr(vae, '_slicing'):
            vae._slicing = None
        if hasattr(vae, '_tiling'):
            vae._tiling = None
            
        # Re-enable standard optimizations
        try:
            vae.enable_slicing()
            vae.enable_tiling()
        except Exception:
            pass
        
        logger.debug("VAE state reset for checkpoint switch")

    def _validate_vae_state(self) -> bool:
        """Validate VAE is in a clean state before generation.
        
        Returns:
            True if VAE state is valid, False otherwise (with warnings logged).
        """
        if self._pipe is None or self._pipe.vae is None:
            logger.warning("VAE validation failed: no pipeline or VAE loaded")
            return False
        
        vae = self._pipe.vae
        valid = True
        
        try:
            param = next(vae.parameters())
            
            # Check device
            if param.device.type != "cuda":
                logger.warning(f"VAE on unexpected device: {param.device}, expected cuda")
                valid = False
            
            # Check dtype (should be float16 or float32)
            if param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                logger.warning(f"VAE has unexpected dtype: {param.dtype}")
                valid = False
            
            # Log current state for debugging
            force_upcast = getattr(vae.config, 'force_upcast', False)
            logger.debug(f"VAE state: device={param.device}, dtype={param.dtype}, force_upcast={force_upcast}")
            
        except StopIteration:
            logger.warning("VAE validation failed: no parameters found")
            valid = False
        except Exception as e:
            logger.warning(f"VAE validation error: {e}")
            valid = False
        
        return valid

    def _validate_device_consistency(self) -> bool:
        """Validate all pipeline components are on CUDA and not on meta device.
        
        Returns:
            True if valid, raises RuntimeError if invalid.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        target_device = torch.device("cuda", 0)
        components = {
            'unet': self._pipe.unet,
            'vae': self._pipe.vae,
            'text_encoder': self._pipe.text_encoder,
            'text_encoder_2': self._pipe.text_encoder_2,
        }
        
        for name, comp in components.items():
            if comp is None:
                continue
            try:
                param = next(comp.parameters())
                if param.device.type == "meta":
                    logger.error(f"{name} has tensors on meta device - pipeline corrupted")
                    raise RuntimeError(f"Pipeline corrupted: {name} on meta device")
                if param.device != target_device:
                    logger.warning(f"{name} on {param.device}, moving to {target_device}")
                    comp.to(target_device)
            except StopIteration:
                pass  # No parameters (unlikely)
            except RuntimeError as e:
                if "meta" in str(e).lower():
                    raise RuntimeError(f"Pipeline corrupted: {name} has meta tensors - {e}")
                raise
        
        return True

    def _get_scheduler(
        self,
        name: str,
        use_karras: bool = False,
        prediction_type: str = "epsilon",
    ):
        """Get scheduler instance by name with options."""
        cfg = self._pipe.scheduler.config
        name_lower = name.lower().strip()
        
        scheduler_cls = SCHEDULER_CLASSES.get(name_lower, EulerAncestralDiscreteScheduler)
        
        kwargs = {}
        
        # V_PREDICTION requires special scheduler settings!
        # This is critical - without these, v_prediction models produce NaN/black images
        if prediction_type == "v_prediction":
            kwargs["prediction_type"] = "v_prediction"
            kwargs["timestep_spacing"] = "trailing"
            kwargs["rescale_betas_zero_snr"] = True
            logger.debug("Scheduler configured for v_prediction: trailing spacing, rescale_betas_zero_snr=True")
        else:
            kwargs["prediction_type"] = "epsilon"
            kwargs["timestep_spacing"] = "leading"
            kwargs["rescale_betas_zero_snr"] = False
        
        if use_karras and name_lower in KARRAS_COMPATIBLE:
            kwargs["use_karras_sigmas"] = True
        
        if name_lower in SDE_SCHEDULERS:
            kwargs["algorithm_type"] = "sde-dpmsolver++"
        
        if name_lower in {"dpm++_2m", "dpmpp_2m", "dpm++_2m_sde", "dpmpp_2m_sde"}:
            kwargs["solver_order"] = 2
        
        return scheduler_cls.from_config(cfg, **kwargs)

    def _calculate_hires_denoising(self, upscale_by: float) -> float:
        """Calculate optimal denoising strength based on upscale factor.
        
        Higher upscale factors need more denoising to add detail.
        Matches old server behavior exactly.
        """
        if upscale_by <= 1.2:
            return 0.45
        elif upscale_by <= 1.5:
            return 0.55
        elif upscale_by <= 1.8:
            return 0.62
        elif upscale_by <= 2.0:
            return 0.68
        else:
            return 0.75

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        use_karras: bool = False,
        prediction_type: Optional[str] = None,
        use_hires_fix: bool = False,
        upscale_by: float = 1.5,
        hires_denoising: Optional[float] = None,
        latent_upscale: bool = True,
        monochrome: bool = False,
        **kwargs,
    ) -> Union[Image.Image, list[Image.Image]]:
        """Generate image with optimized VRAM usage.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Number of inference steps
            cfg_scale: CFG guidance scale
            seed: Random seed for reproducibility
            scheduler: Scheduler name
            use_karras: Use karras sigmas
            prediction_type: Override detected prediction type
            use_hires_fix: Enable hires fix (upscale + refine)
            upscale_by: Hires upscale factor (default 1.5x)
            hires_denoising: Override auto-calculated denoising strength
            latent_upscale: Use fast latent upscaling (default True, ~1.5s faster)
            monochrome: Convert output to grayscale (black and white)
            **kwargs: Additional arguments
            
        Returns:
            Generated PIL image
        """
        self.touch()
        
        # Validate device consistency before generation
        self._validate_device_consistency()
        
        # Validate VAE state to catch dtype poisoning early
        self._validate_vae_state()

        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Apply defaults
        width = width or self._config.default_width
        height = height or self._config.default_height
        steps = steps or self._config.default_steps
        cfg_scale = cfg_scale or self._config.default_cfg_scale
        scheduler_name = scheduler or self._config.default_scheduler
        
        # Use detected prediction type if not explicitly provided
        actual_prediction_type = prediction_type or self._prediction_type

        # Ensure VAE is in correct state (prevents NaN after checkpoint switch)
        self._ensure_vae_ready()

        # Set scheduler with correct prediction type (creates fresh scheduler)
        self._pipe.scheduler = self._get_scheduler(
            scheduler_name,
            use_karras=use_karras,
            prediction_type=actual_prediction_type,
        )
        # Refiner shares scheduler with main pipe
        self._refiner_pipe.scheduler = self._pipe.scheduler

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.debug(
            f"Generating: {width}x{height}, {steps} steps, cfg={cfg_scale}, "
            f"scheduler={scheduler_name}, prediction_type={actual_prediction_type}"
        )

        # Pre-generation cleanup
        torch.cuda.empty_cache()

        gen_start = time.time()

        # Generate with inference mode
        with torch.inference_mode():
            # === PASS 1: Text-to-Image ===
            # Encode prompts ONCE (reuse for hires pass)
            # Use tokenize_and_check for efficient caching (avoids double tokenization)
            _t_encode = time.time()
            prompt_is_long, prompt_cached = tokenize_and_check(self._pipe, prompt)
            neg_is_long, neg_cached = tokenize_and_check(self._pipe, negative_prompt or "")
            use_long_prompt = prompt_is_long or neg_is_long
            
            if use_long_prompt:
                logger.debug("Using long prompt encoding (prompt exceeds 77 tokens)")
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    _prompt_attention_mask,
                    _negative_attention_mask,
                ) = encode_long_prompt(
                    pipe=self._pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    device=torch.device("cuda"),
                    cached_tokens={
                        'prompt': prompt_cached,
                        'negative_prompt': neg_cached
                    }
                )
            else:
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self._pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=cfg_scale > 1.0,
                )
            
            logger.debug(f"Running Pass 1 (Txt2Img) - {width}x{height}...")
            
            _t_pass1 = time.time()
            
            # Get latents with pre-encoded prompts
            latents = self._pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="latent",  # Return latents for manual VAE decode
                callback_on_step_end=self._create_abort_callback(),
            ).images
            
            _t_pass2 = time.time()
            logger.info(
                f"[timing] prompt_encode={_t_pass1 - _t_encode:.2f}s, "
                f"pass1_denoise={_t_pass2 - _t_pass1:.2f}s"
            )
            
            # === PASS 2: Hires Fix (if enabled) ===
            if use_hires_fix and self._refiner_pipe is not None:
                new_width = int(width * upscale_by)
                new_height = int(height * upscale_by)
                
                # Light VRAM cleanup only - skip gc.collect() and synchronize() for speed
                torch.cuda.empty_cache()
                
                # Calculate denoising strength (auto or override)
                denoising = hires_denoising if hires_denoising is not None else self._calculate_hires_denoising(upscale_by)
                
                logger.info(f"Running Pass 2 (Hires Fix) - {width}x{height} -> {new_width}x{new_height} (strength: {denoising:.2f}, latent_upscale: {latent_upscale})")
                
                if latent_upscale:
                    # FAST PATH: Upscale latents directly (~1.5s faster)
                    # Skip: decode->upscale->encode by doing everything in latent space
                    # Uses bicubic (sharper) + small noise to preserve detail
                    upscaled_latents = _upscale_latents(
                        latents, 
                        scale_factor=upscale_by, 
                        mode='bicubic',
                        antialiasing_noise=0.02  # 2% noise restores high-freq detail
                    )
                    
                    # Safety check for NaN
                    if torch.isnan(upscaled_latents).any():
                        logger.warning("NaN detected in upscaled latents, falling back to image upscaling")
                        latent_upscale = False  # Force slow path below
                    else:
                        # Pass pre-upscaled latents directly to img2img
                        # The latents parameter bypasses the VAE encode step entirely
                        # We still need a dummy image for sizing, but it won't be encoded
                        hires_latents = self._refiner_pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            image=Image.new('RGB', (new_width, new_height)),  # Dummy for sizing
                            latents=upscaled_latents,  # Pre-upscaled latents bypass encode
                            num_inference_steps=steps,
                            strength=denoising,
                            guidance_scale=cfg_scale,
                            generator=generator,
                            output_type="latent",
                            callback_on_step_end=self._create_abort_callback(),
                        ).images
                        
                # SLOW PATH: Decode->Upscale->Encode (original method or NaN fallback)
                if not latent_upscale:
                    # Decode latents to image for upscaling
                    image = _decode_latents_to_image(self._pipe.vae, latents)
                    
                    # Upscale with Lanczos interpolation (high quality)
                    image = image.resize((new_width, new_height), resample=Image.LANCZOS)
                    
                    # Run img2img refinement with SAME prompt embeddings (no re-encoding)
                    hires_latents = self._refiner_pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                        image=image,
                        num_inference_steps=steps,
                        strength=denoising,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        output_type="latent",
                        callback_on_step_end=self._create_abort_callback(),
                    ).images
                
                # Decode hires latents
                image = _decode_latents_to_image(self._pipe.vae, hires_latents)
            else:
                # No hires fix - just decode the latents
                image = _decode_latents_to_image(self._pipe.vae, latents)

        logger.info(
            f"[timing] hires+vae_decode={time.time() - _t_pass2:.2f}s, "
            f"total_gpu={time.time() - gen_start:.2f}s"
        )

        # Note: _decode_latents_to_image already synchronizes before returning
        # No need for redundant sync here - it would stall the pipeline
        
        # Post-generation cleanup (light)
        torch.cuda.empty_cache()

        # Convert to grayscale if monochrome is requested
        if monochrome:
            image = image.convert("L").convert("RGB")
            logger.debug("Converted image to monochrome")

        return image

    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        use_karras: bool = False,
        prediction_type: Optional[str] = None,
        monochrome: bool = False,
        **kwargs,
    ) -> Image.Image:
        """Refine an input image using the img2img pipeline.

        Args:
            image: Input PIL image (must be RGB-convertible)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            strength: Denoising strength (0.0 = no change, 1.0 = full redraw)
            steps: Number of inference steps
            cfg_scale: CFG guidance scale
            seed: Random seed for reproducibility
            scheduler: Scheduler name
            use_karras: Use karras sigmas
            prediction_type: Override detected prediction type
            monochrome: Convert output to grayscale (black and white)
            **kwargs: Additional arguments

        Returns:
            Generated PIL image
        """
        self.touch()

        # Validate device consistency before generation
        self._validate_device_consistency()

        # Validate VAE state to catch dtype poisoning early
        self._validate_vae_state()

        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        if self._refiner_pipe is None:
            raise RuntimeError("img2img pipeline unavailable (refiner not initialized).")

        if not 0.0 < strength < 1.0:
            raise ValueError("strength must be between 0.0 and 1.0 (exclusive).")

        # Apply defaults
        steps = steps or self._config.default_steps
        cfg_scale = cfg_scale or self._config.default_cfg_scale
        scheduler_name = scheduler or self._config.default_scheduler

        # Use detected prediction type if not explicitly provided
        actual_prediction_type = prediction_type or self._prediction_type

        # Ensure VAE is in correct state (prevents NaN after checkpoint switch)
        self._ensure_vae_ready()

        # Normalize input image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Set scheduler with correct prediction type (creates fresh scheduler)
        self._pipe.scheduler = self._get_scheduler(
            scheduler_name,
            use_karras=use_karras,
            prediction_type=actual_prediction_type,
        )
        # Refiner shares scheduler with main pipe
        self._refiner_pipe.scheduler = self._pipe.scheduler

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.debug(
            f"Running img2img - {image.width}x{image.height}, strength={strength}, "
            f"{steps} steps, cfg={cfg_scale}, scheduler={scheduler_name}, "
            f"prediction_type={actual_prediction_type}"
        )

        # Pre-generation cleanup
        torch.cuda.empty_cache()

        gen_start = time.time()

        # Refiner shares all components with the main pipe - no extra VRAM
        with torch.inference_mode():
            latents = self._refiner_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                output_type="latent",
                callback_on_step_end=self._create_abort_callback(),
            ).images

            result = _decode_latents_to_image(self._refiner_pipe.vae, latents)

        logger.info(f"[timing] img2img total_gpu={time.time() - gen_start:.2f}s")

        # Post-generation cleanup (light)
        torch.cuda.empty_cache()

        # Convert to grayscale if monochrome is requested
        if monochrome:
            result = result.convert("L").convert("RGB")
            logger.debug("Converted image to monochrome")

        return result

    def estimate_vram(self) -> int:
        """Estimate VRAM needed (~7GB for SDXL)."""
        return 7 * 1024 ** 3
