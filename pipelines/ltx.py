"""LTX Video Pipeline implementation for video generation."""

import gc
import inspect
import io
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from PIL import Image

from .base import BasePipeline, pipeline_module
from ..lora import LoRAManager
from ..optimizations.cuda_streams import get_stream_manager

logger = logging.getLogger(__name__)


@dataclass
class LTXGuiderParams:
    """Multi-modal guidance parameters for LTX-2 video generation.
    
    These parameters control classifier-free guidance (CFG) and spatio-temporal
    guidance (STG) for improved temporal coherence and quality.
    
    Attributes:
        cfg_scale: Classifier-Free Guidance scale. Higher values follow prompt more
                   closely but may reduce quality. Default 3.0 is balanced.
        stg_scale: Spatio-Temporal Guidance scale for temporal coherence. Higher
                   values improve frame-to-frame consistency. Default 1.0.
        stg_blocks: List of transformer block indices to perturb for STG.
                    Block 29 is typically most effective for temporal coherence.
        rescale_scale: CFG rescale factor to prevent over-saturation. Values < 1.0
                       reduce the variance boost from guidance. Default 0.7.
    """
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    stg_blocks: List[int] = field(default_factory=lambda: [29])
    rescale_scale: float = 0.7


@dataclass
class RestartSamplerParams:
    """Parameters for Restart sampling to improve video motion quality.
    
    Restart sampling (arXiv:2306.14878) improves generative quality by adding
    controlled noise at intermediate points during denoising. This helps produce
    more dynamic motion in video generation.
    
    The scheduler's step() method supports stochastic parameters (s_churn, s_tmin,
    s_tmax, s_noise) which enable restart-like behavior in Flow Matching.
    
    Attributes:
        num_segments: Number of restart segments (2 for Res_2s, 3 for Res_3s).
                      More segments = more noise reinjection = potentially more motion.
        restart_strength: Amount of noise to add at restart points (0.0-1.0).
                         Higher values add more noise for stronger restart effect.
        s_churn: Stochasticity parameter for the scheduler step (default 0.0).
                 Higher values add more randomness during each step.
        s_tmin: Minimum sigma for applying stochasticity (default 0.0).
        s_tmax: Maximum sigma for applying stochasticity (default inf).
        s_noise: Scale factor for noise addition (default 1.0).
    """
    num_segments: int = 2  # Default: 2 segments (Res_2s)
    restart_strength: float = 0.3  # Conservative default for video
    s_churn: float = 0.0  # Scheduler stochasticity
    s_tmin: float = 0.0
    s_tmax: float = float('inf')
    s_noise: float = 1.0


@pipeline_module(
    name="ltx",
    display_name="LTX Video Generation",
    output_type="video",
    conflicts_with=["sdxl", "zimg", "tts"],  # LTX is large, conflicts with everything
    checkpoint_patterns=[r"ltx", r"lightricks"],
    vram_estimate_gb=15.0,
    supports_checkpoints=False,
)
class LTXPipeline(BasePipeline):
    """LTX Video pipeline with VRAM management.
    
    Supports both text-to-video and image-to-video generation modes.
    
    FP8 Quantization:
        Enable fp8_enabled=True to reduce VRAM usage from ~15GB to ~8GB.
        For best results, set environment variable before running:
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    """

    # Model ID for HuggingFace
    MODEL_ID = "Lightricks/LTX-Video"
    
    # Default negative prompt to prevent still/static video output
    DEFAULT_MOTION_NEGATIVE = "still image, still video, no motion, static, frozen, motionless"

    def __init__(self, mode: str = "text-to-video", fp8_enabled: bool = False):
        """Initialize LTX pipeline.
        
        Args:
            mode: Generation mode - "text-to-video" or "image-to-video"
            fp8_enabled: Enable FP8 quantization for reduced VRAM (~8GB vs ~15GB)
        """
        self._pipe = None
        self._parked = False
        self._mode = mode
        self._device = "cpu"
        self._last_had_loras = False
        self._fp8_enabled = fp8_enabled

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None and self._device == "cuda"

    @property
    def is_parked(self) -> bool:
        return self._pipe is not None and self._device == "cpu"

    def load(self) -> None:
        """Load LTX pipeline to CUDA."""
        if self.is_loaded:
            logger.debug("LTX already loaded, skipping")
            return

        logger.info(f"Loading LTX Video pipeline (mode={self._mode})")

        # Clear memory before loading
        gc.collect()
        torch.cuda.empty_cache()

        if self._mode == "image-to-video":
            from diffusers import LTXImageToVideoPipeline
            self._pipe = LTXImageToVideoPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            from diffusers import LTXPipeline as DiffusersLTXPipeline
            self._pipe = DiffusersLTXPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        self._pipe.to("cuda")
        self._device = "cuda"
        self._parked = False

        # Enable memory optimizations
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            logger.debug("LTX xformers enabled")
        except Exception:
            logger.debug("LTX xformers not available")

        # Apply FP8 quantization if enabled
        self._enable_fp8_inference()

        logger.info(f"LTX Video pipeline loaded (mode={self._mode}, fp8={self._fp8_enabled})")

    def unload(self) -> None:
        """Unload pipeline and free VRAM."""
        if self._pipe is None:
            return

        logger.info("Unloading LTX Video pipeline")

        # Clean up LoRA state before unload
        try:
            LoRAManager().prepare_for_checkpoint_switch(self._pipe)
        except Exception as e:
            logger.warning(f"LoRA cleanup failed during unload: {e}")
        finally:
            LoRAManager().reset_loaded_adapters()

        try:
            self._pipe.to("cpu")
        except Exception:
            pass

        del self._pipe
        self._pipe = None
        self._device = "cpu"
        self._parked = False
        self._last_had_loras = False

        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("LTX Video unloaded")

    def to_cpu(self) -> None:
        """Park pipeline to CPU."""
        if self._pipe is None or self._parked:
            return

        logger.info("Parking LTX Video to CPU")
        self._pipe.to("cpu")
        self._device = "cpu"
        self._parked = True

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("LTX Video parked")

    def to_gpu(self) -> None:
        """Restore pipeline to GPU."""
        if self._pipe is None or not self._parked:
            return

        logger.info("Restoring LTX Video to GPU")
        self._pipe.to("cuda")
        self._device = "cuda"
        self._parked = False

        logger.info("LTX Video restored")

    def estimate_vram(self) -> int:
        """Estimated VRAM usage (~15GB, or ~8GB with FP8 enabled)."""
        if self._fp8_enabled:
            return 8 * 1024 ** 3
        return 15 * 1024 ** 3

    def _enable_fp8_inference(self):
        """Enable FP8 inference for reduced memory footprint."""
        if not self._fp8_enabled or self._pipe is None:
            return
        
        # Check if transformer supports FP8
        if hasattr(self._pipe, 'transformer'):
            try:
                # Cast transformer weights to FP8 for storage, upcast during inference
                for name, param in self._pipe.transformer.named_parameters():
                    if param.dtype == torch.bfloat16:
                        # Store in FP8 format
                        param.data = param.data.to(torch.float8_e4m3fn)
                logger.info("FP8 quantization enabled for transformer")
            except Exception as e:
                logger.warning(f"FP8 quantization failed: {e}")

    def _calculate_scaled_dimensions(
        self,
        orig_width: int,
        orig_height: int,
        target_avg: int = 768,
        divisor: int = 32,
    ) -> tuple:
        """Scale dimensions to be divisible by divisor with average closest to target_avg.
        
        Preserves aspect ratio. Default 768 is more stable for LTX video generation.
        
        Args:
            orig_width: Original image width
            orig_height: Original image height
            target_avg: Target average dimension (default 768)
            divisor: Dimension must be divisible by this (default 32)
        
        Returns:
            Tuple of (new_width, new_height)
        """
        aspect = orig_width / orig_height

        # Find scale factor where (w + h) / 2 = target_avg
        # w = aspect * h, so (aspect * h + h) / 2 = target_avg
        # h * (aspect + 1) / 2 = target_avg
        # h = 2 * target_avg / (aspect + 1)
        target_h = 2 * target_avg / (aspect + 1)
        target_w = aspect * target_h

        # Round to nearest divisor
        new_w = round(target_w / divisor) * divisor
        new_h = round(target_h / divisor) * divisor

        # Ensure minimum size
        new_w = max(divisor, new_w)
        new_h = max(divisor, new_h)

        return new_w, new_h

    def _preprocess_image_for_video(self, image: Image.Image, img_compression: int = 42) -> Image.Image:
        """Apply compression preprocessing to help LTX recognize image as video frame.
        
        CRITICAL: LTX-Video was trained on compressed video frames. Clean PNGs cause
        "still video" output because LTX treats them as photographs.
        
        This matches ComfyUI's LTXVPreprocess node behavior.
        
        Args:
            image: Input PIL Image
            img_compression: Compression level 0-100 (higher = more compression, default 42)
                            Maps to JPEG quality as (100 - img_compression * 0.65)
        
        Returns:
            Image with compression artifacts that LTX recognizes as video-like
        """
        # Convert ComfyUI compression (0-100, higher=more) to JPEG quality (0-100, lower=more)
        # Based on testing, img_compression=42 should give moderate compression
        # Formula: quality = 100 - (compression * 0.65) gives quality ~73 for compression=42
        jpeg_quality = max(10, min(95, int(100 - img_compression * 0.65)))
        
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        result = Image.open(buffer).convert('RGB')
        
        logger.debug(f"Preprocessed image: compression={img_compression} -> JPEG quality={jpeg_quality}")
        return result

    def _decode_latent_tiles(
        self,
        latents: torch.Tensor,
        timestep_tensor: Optional[torch.Tensor],
        spatial_tile_size: int = 256,
        temporal_tile_size: int = 8,
        tile_overlap: float = 0.25,
    ) -> torch.Tensor:
        """Decode latents using spatio-temporal tiling for memory efficiency.
        
        Processes video in both spatial (H, W) and temporal (F) chunks,
        blending overlapping regions for smooth output.
        
        Args:
            latents: Latent tensor [B, C, F, H, W]
            timestep_tensor: Timestep for VAE decode
            spatial_tile_size: Size of spatial tiles in latent space
            temporal_tile_size: Number of frames per temporal tile
            tile_overlap: Overlap ratio between tiles (0-0.5)
        
        Returns:
            Decoded video tensor
        """
        B, C, F, H, W = latents.shape
        
        # If latents are small enough, decode directly
        if H <= spatial_tile_size and W <= spatial_tile_size and F <= temporal_tile_size:
            logger.debug(f"Latents small enough for direct decode: {H}x{W}x{F}")
            return self._pipe.vae.decode(latents, timestep_tensor, return_dict=False)[0]
        
        logger.info(f"Using spatio-temporal tiled decode: latent={H}x{W}x{F}, "
                   f"tile_size={spatial_tile_size}, temporal_tile={temporal_tile_size}, overlap={tile_overlap}")
        
        # Calculate overlaps
        spatial_overlap = max(1, int(spatial_tile_size * tile_overlap))
        temporal_overlap = max(1, int(temporal_tile_size * tile_overlap))
        
        # Calculate step sizes (tile_size - overlap)
        spatial_step = spatial_tile_size - spatial_overlap
        temporal_step = temporal_tile_size - temporal_overlap
        
        # Calculate number of tiles needed
        n_tiles_h = max(1, (H - spatial_overlap + spatial_step - 1) // spatial_step) if H > spatial_tile_size else 1
        n_tiles_w = max(1, (W - spatial_overlap + spatial_step - 1) // spatial_step) if W > spatial_tile_size else 1
        n_tiles_t = max(1, (F - temporal_overlap + temporal_step - 1) // temporal_step) if F > temporal_tile_size else 1
        
        logger.debug(f"Tile grid: {n_tiles_h}x{n_tiles_w}x{n_tiles_t} tiles (H x W x T)")
        
        # Get VAE scale factor (typically 8 for spatial)
        vae_spatial_scale = getattr(self._pipe, 'vae_spatial_compression_ratio', 32) // 4  # Approx output scale
        vae_temporal_scale = getattr(self._pipe, 'vae_temporal_compression_ratio', 8)
        
        # Output dimensions (approximate - VAE may adjust)
        out_h = H * vae_spatial_scale
        out_w = W * vae_spatial_scale
        out_f = (F - 1) * vae_temporal_scale + 1  # LTX temporal unpacking
        
        # Initialize output tensor and weight accumulator
        output = torch.zeros((B, 3, out_f, out_h, out_w), device=latents.device, dtype=latents.dtype)
        weight = torch.zeros((B, 1, out_f, out_h, out_w), device=latents.device, dtype=latents.dtype)
        
        # Create blending weights (cosine ramp for smooth transitions)
        def create_blend_weight(size: int, overlap: int) -> torch.Tensor:
            """Create 1D cosine blend weight."""
            w = torch.ones(size, device=latents.device, dtype=latents.dtype)
            if overlap > 0 and size > overlap * 2:
                # Ramp up at start
                ramp = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, overlap, device=latents.device, dtype=latents.dtype)))
                w[:overlap] = ramp
                # Ramp down at end
                w[-overlap:] = ramp.flip(0)
            return w
        
        total_tiles = n_tiles_t * n_tiles_h * n_tiles_w
        tile_idx = 0
        
        # Process tiles
        for t_idx in range(n_tiles_t):
            t_start = min(t_idx * temporal_step, F - temporal_tile_size) if F > temporal_tile_size else 0
            t_end = min(t_start + temporal_tile_size, F)
            
            for h_idx in range(n_tiles_h):
                h_start = min(h_idx * spatial_step, H - spatial_tile_size) if H > spatial_tile_size else 0
                h_end = min(h_start + spatial_tile_size, H)
                
                for w_idx in range(n_tiles_w):
                    w_start = min(w_idx * spatial_step, W - spatial_tile_size) if W > spatial_tile_size else 0
                    w_end = min(w_start + spatial_tile_size, W)
                    
                    tile_idx += 1
                    logger.debug(f"Decoding tile {tile_idx}/{total_tiles}: "
                               f"t[{t_start}:{t_end}] h[{h_start}:{h_end}] w[{w_start}:{w_end}]")
                    
                    # Extract tile
                    tile_latent = latents[:, :, t_start:t_end, h_start:h_end, w_start:w_end].contiguous()
                    
                    # Decode tile
                    with torch.no_grad():
                        tile_decoded = self._pipe.vae.decode(tile_latent, timestep_tensor, return_dict=False)[0]
                    
                    # Get actual decoded dimensions
                    _, _, dec_t, dec_h, dec_w = tile_decoded.shape
                    
                    # Calculate output positions
                    out_t_start = t_start * vae_temporal_scale if t_start > 0 else 0
                    out_t_end = out_t_start + dec_t
                    out_h_start = h_start * vae_spatial_scale
                    out_h_end = out_h_start + dec_h
                    out_w_start = w_start * vae_spatial_scale
                    out_w_end = out_w_start + dec_w
                    
                    # Clamp to output size
                    out_t_end = min(out_t_end, out_f)
                    out_h_end = min(out_h_end, out_h)
                    out_w_end = min(out_w_end, out_w)
                    
                    # Create blend weights for this tile
                    blend_t = create_blend_weight(dec_t, temporal_overlap * vae_temporal_scale)
                    blend_h = create_blend_weight(dec_h, spatial_overlap * vae_spatial_scale)
                    blend_w = create_blend_weight(dec_w, spatial_overlap * vae_spatial_scale)
                    
                    # Combine into 3D weight
                    tile_weight = blend_t[None, None, :dec_t, None, None] * \
                                 blend_h[None, None, None, :dec_h, None] * \
                                 blend_w[None, None, None, None, :dec_w]
                    
                    # Trim decoded tile to fit
                    actual_t = out_t_end - out_t_start
                    actual_h = out_h_end - out_h_start
                    actual_w = out_w_end - out_w_start
                    
                    # Accumulate weighted output
                    output[:, :, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end] += \
                        tile_decoded[:, :, :actual_t, :actual_h, :actual_w] * tile_weight[:, :, :actual_t, :actual_h, :actual_w]
                    weight[:, :, out_t_start:out_t_end, out_h_start:out_h_end, out_w_start:out_w_end] += \
                        tile_weight[:, :, :actual_t, :actual_h, :actual_w]
                    
                    # Clear tile from memory
                    del tile_decoded, tile_latent, tile_weight
                    torch.cuda.empty_cache()
        
        # Normalize by accumulated weights
        output = output / (weight + 1e-8)
        
        logger.info(f"Spatio-temporal tiled decode complete: {total_tiles} tiles processed")
        return output

    def _smart_decode(
        self,
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        decode_timestep: float = 0.05,
        decode_noise_scale: Optional[float] = None,
        spatial_tile_size: int = 256,
        temporal_tile_size: int = 8,
        tile_overlap: float = 0.25,
    ) -> List[Image.Image]:
        """Decode latents with memory optimization using spatio-temporal tiling.
        
        Offloads transformer to CPU and uses tiled VAE decoding to prevent VRAM overflow.
        Supports both spatial (H, W) and temporal (F) tiling for large videos.
        
        Args:
            latents: Latent tensor from pipeline (packed format [B, S, D])
            num_frames: Number of video frames
            height: Video height in pixels
            width: Video width in pixels
            decode_timestep: Timestep for VAE decode (default 0.05)
            decode_noise_scale: Noise scale for decode (default same as decode_timestep)
            spatial_tile_size: Size of spatial tiles in latent space (default 256)
            temporal_tile_size: Number of frames per temporal tile (default 8)
            tile_overlap: Overlap ratio between tiles (default 0.25)
        
        Returns:
            List of PIL Image frames
        """
        logger.info("Smart decode: offloading transformer to CPU...")

        # Move transformer to CPU to free VRAM for VAE decode
        if hasattr(self._pipe, 'transformer'):
            self._pipe.transformer.to("cpu")

        # Also move text encoders to CPU
        if hasattr(self._pipe, 'text_encoder'):
            self._pipe.text_encoder.to("cpu")
        if hasattr(self._pipe, 'text_encoder_2'):
            self._pipe.text_encoder_2.to("cpu")

        # Clear VRAM
        torch.cuda.empty_cache()
        gc.collect()

        # Configure VAE tiling for spatio-temporal decode
        use_builtin_tiling = False
        if hasattr(self._pipe, 'vae'):
            if hasattr(self._pipe.vae, 'enable_tiling'):
                self._pipe.vae.enable_tiling()
                use_builtin_tiling = True
                logger.debug("VAE built-in tiling enabled")
                
                # Configure tile sizes if VAE supports it
                if hasattr(self._pipe.vae, 'tile_latent_min_size'):
                    self._pipe.vae.tile_latent_min_size = spatial_tile_size
                    logger.debug(f"VAE tile_latent_min_size set to {spatial_tile_size}")
                if hasattr(self._pipe.vae, 'tile_sample_min_size'):
                    # Sample size is typically spatial_tile_size * vae_scale_factor
                    vae_scale = getattr(self._pipe.vae.config, 'scaling_factor', 8)
                    self._pipe.vae.tile_sample_min_size = spatial_tile_size * vae_scale
                    logger.debug(f"VAE tile_sample_min_size set to {spatial_tile_size * vae_scale}")
                if hasattr(self._pipe.vae, 'tile_overlap_factor'):
                    self._pipe.vae.tile_overlap_factor = tile_overlap
                    logger.debug(f"VAE tile_overlap_factor set to {tile_overlap}")
            
            if hasattr(self._pipe.vae, 'enable_slicing'):
                self._pipe.vae.enable_slicing()
                logger.debug("VAE slicing enabled")
        
        logger.info(f"Spatio-temporal decode config: spatial_tile={spatial_tile_size}, "
                   f"temporal_tile={temporal_tile_size}, overlap={tile_overlap}, "
                   f"builtin_tiling={'enabled' if use_builtin_tiling else 'disabled'}")

        # Move VAE to GPU and ensure correct dtype
        self._pipe.vae.to("cuda")
        vae_dtype = next(self._pipe.vae.parameters()).dtype

        logger.debug(f"Input latent shape: {latents.shape}, dtype: {latents.dtype}")

        # Calculate latent dimensions using pipeline compression ratios
        vae_temporal_ratio = getattr(self._pipe, 'vae_temporal_compression_ratio', 8)
        vae_spatial_ratio = getattr(self._pipe, 'vae_spatial_compression_ratio', 32)
        
        latent_num_frames = (num_frames - 1) // vae_temporal_ratio + 1
        latent_height = height // vae_spatial_ratio
        latent_width = width // vae_spatial_ratio

        # LTX returns packed latents [B, S, D] when output_type="latent"
        # We need to unpack to [B, C, F, H, W] for VAE decode
        if latents.dim() == 3:
            # Get patch sizes from pipeline (default to 1 for LTX-Video)
            spatial_patch_size = getattr(self._pipe, 'transformer_spatial_patch_size', 1)
            temporal_patch_size = getattr(self._pipe, 'transformer_temporal_patch_size', 1)
            
            logger.debug(f"Unpacking latents: frames={latent_num_frames}, h={latent_height}, w={latent_width}, "
                        f"spatial_patch={spatial_patch_size}, temporal_patch={temporal_patch_size}")
            
            # Unpack latents from [B, S, D] to [B, C, F, H, W]
            latents = self._pipe._unpack_latents(
                latents,
                latent_num_frames,
                latent_height,
                latent_width,
                spatial_patch_size,
                temporal_patch_size,
            )
            logger.debug(f"Unpacked latent shape: {latents.shape}")
        
        # Denormalize latents
        latents = self._pipe._denormalize_latents(
            latents,
            self._pipe.vae.latents_mean,
            self._pipe.vae.latents_std,
            self._pipe.vae.config.scaling_factor,
        )
        logger.debug("Latents denormalized")

        # Convert latents to match VAE dtype and move to GPU
        latents = latents.to(device="cuda", dtype=vae_dtype)

        # CRITICAL: Add decode noise for quality (matches diffusers pipeline behavior)
        # This prevents the extremely blurry output
        if decode_noise_scale is None:
            decode_noise_scale = decode_timestep
        
        if decode_timestep > 0:
            batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            timestep_tensor = torch.tensor([decode_timestep] * batch_size, device="cuda", dtype=latents.dtype)
            noise_scale_tensor = torch.tensor([decode_noise_scale] * batch_size, device="cuda", dtype=latents.dtype)
            noise_scale_tensor = noise_scale_tensor[:, None, None, None, None]  # Reshape for broadcasting
            latents = (1 - noise_scale_tensor) * latents + noise_scale_tensor * noise
            logger.debug(f"Added decode noise: timestep={decode_timestep}, noise_scale={decode_noise_scale}")
        else:
            timestep_tensor = None

        logger.debug(f"Final latent shape: {latents.shape}, dtype: {latents.dtype}")

        # Convert to VAE dtype for decode
        latents = latents.to(self._pipe.vae.dtype)

        # Decode using the VAE with overlapped prefetch
        logger.info("Decoding latents...")
        decode_start = time.perf_counter()
        
        # Try to use CUDA stream prefetching for overlapped execution
        stream_mgr = None
        prefetch_started = False
        try:
            stream_mgr = get_stream_manager()
            
            # Start prefetch of transformer on copy stream while VAE decodes on default stream
            # This overlaps the H2D transfer with VAE decode compute
            if hasattr(self._pipe, 'transformer') and self._pipe.transformer is not None:
                prefetch_start = time.perf_counter()
                with stream_mgr.copy_context():
                    self._pipe.transformer.to("cuda", non_blocking=True)
                prefetch_started = True
                logger.debug(f"Transformer prefetch started on copy stream (took {(time.perf_counter() - prefetch_start)*1000:.1f}ms to initiate)")
        except Exception as e:
            logger.warning(f"CUDA stream manager not available, falling back to sequential execution: {e}")
            stream_mgr = None
        
        # VAE decode runs on default stream (overlaps with prefetch on copy stream)
        vae_decode_start = time.perf_counter()
        
        # Check if we need custom spatio-temporal tiling (for larger videos)
        B, C, F, H, W = latents.shape
        use_custom_tiling = (H > spatial_tile_size or W > spatial_tile_size or F > temporal_tile_size)
        
        with torch.no_grad():
            if use_custom_tiling and not use_builtin_tiling:
                # Use our custom spatio-temporal tiling for large videos without built-in support
                logger.info(f"Using custom spatio-temporal tiling: latent={H}x{W}x{F}")
                video = self._decode_latent_tiles(
                    latents, 
                    timestep_tensor,
                    spatial_tile_size=spatial_tile_size,
                    temporal_tile_size=temporal_tile_size,
                    tile_overlap=tile_overlap,
                )
            else:
                # Use VAE's built-in decode (with configured tiling if available)
                video = self._pipe.vae.decode(latents, timestep_tensor, return_dict=False)[0]
        
        vae_decode_time = (time.perf_counter() - vae_decode_start) * 1000

        # Wait for prefetch to complete before continuing
        if stream_mgr is not None and prefetch_started:
            prefetch_wait_start = time.perf_counter()
            stream_mgr.wait_for_copy()
            prefetch_wait_time = (time.perf_counter() - prefetch_wait_start) * 1000
            logger.info(f"VAE decode: {vae_decode_time:.1f}ms, prefetch wait: {prefetch_wait_time:.1f}ms (overlapped)")
        else:
            logger.info(f"VAE decode: {vae_decode_time:.1f}ms (sequential)")

        logger.info(f"Decode complete in {(time.perf_counter() - decode_start)*1000:.1f}ms total")
        
        # Use the pipeline's video processor for proper postprocessing
        frames = self._pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        
        # CRITICAL: Restore pipeline state after smart decode
        # Guard against race condition where pipeline was unloaded during decode
        if self._pipe is not None:
            # Disable VAE tiling - it uses slow_conv3d which only works on CPU
            # This would break VAE encode for image-to-video on the next request
            if hasattr(self._pipe, 'vae') and self._pipe.vae is not None:
                if hasattr(self._pipe.vae, 'disable_tiling'):
                    self._pipe.vae.disable_tiling()
                    logger.debug("VAE tiling disabled")
                if hasattr(self._pipe.vae, 'disable_slicing'):
                    self._pipe.vae.disable_slicing()
                    logger.debug("VAE slicing disabled")
            
            # Restore text encoders to CUDA (transformer already prefetched if stream manager available)
            restore_start = time.perf_counter()
            
            # Only restore transformer here if prefetch didn't happen
            if not prefetch_started:
                if hasattr(self._pipe, 'transformer') and self._pipe.transformer is not None:
                    self._pipe.transformer.to("cuda")
                    logger.debug("Transformer restored to CUDA (sequential)")
            
            if hasattr(self._pipe, 'text_encoder') and self._pipe.text_encoder is not None:
                self._pipe.text_encoder.to("cuda")
            if hasattr(self._pipe, 'text_encoder_2') and self._pipe.text_encoder_2 is not None:
                self._pipe.text_encoder_2.to("cuda")
            
            restore_time = (time.perf_counter() - restore_start) * 1000
            logger.debug(f"Pipeline components restored to CUDA in {restore_time:.1f}ms")
        else:
            logger.warning("Pipeline was unloaded during decode, skipping restoration")
        
        return frames

    def _gradient_estimating_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
        prev_pred: Optional[torch.Tensor],
        sigma_t: float,
        ge_gamma: float = 2.0,
    ) -> tuple:
        """Perform a gradient estimation denoising step for faster inference.
        
        Instead of computing the full denoising step, estimates the gradient from
        previous steps. This allows using 20-30 steps instead of 40-50 while
        maintaining quality.
        
        Formula: x_{t-1} = x_t - sigma_t * (cond_pred - (1 + ge_gamma) * uncond_pred + ge_gamma * prev_pred)
        
        Args:
            latents: Current latent tensor x_t
            timestep: Current timestep
            cond_pred: Conditional prediction (with prompt)
            uncond_pred: Unconditional prediction (without prompt)
            prev_pred: Previous step's prediction (None for first step)
            sigma_t: Noise level at timestep t
            ge_gamma: Gradient estimation coefficient (default 2.0)
        
        Returns:
            Tuple of (denoised latents x_{t-1}, current prediction for next step)
        """
        if prev_pred is None:
            # First step: use standard CFG
            # noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            noise_pred = cond_pred
            denoised = latents - sigma_t * noise_pred
        else:
            # Gradient estimation step
            # x_{t-1} = x_t - sigma_t * (cond_pred - (1 + ge_gamma) * uncond_pred + ge_gamma * prev_pred)
            noise_pred = cond_pred - (1 + ge_gamma) * uncond_pred + ge_gamma * prev_pred
            denoised = latents - sigma_t * noise_pred
        
        # Return current conditional prediction as prev_pred for next step
        return denoised, cond_pred

    def _tensor_to_pil_frames(self, video_tensor: torch.Tensor) -> List[Image.Image]:
        """Convert video tensor to list of PIL images.
        
        Args:
            video_tensor: Video tensor in various formats:
                - [B, C, F, H, W] - 5D from VAE decode (squeeze batch)
                - [C, F, H, W] - 4D channels first
                - [F, C, H, W] - 4D frames first
                - [F, H, W, C] - 4D already in correct format
        
        Returns:
            List of PIL Image frames
        """
        import numpy as np
        video = video_tensor
        
        logger.debug(f"Input video tensor shape: {video.shape}, dtype: {video.dtype}")

        # Handle 5D tensor [B, C, F, H, W] - squeeze batch dimension
        if video.dim() == 5:
            video = video.squeeze(0)  # -> [C, F, H, W]
            logger.debug(f"Squeezed to 4D: {video.shape}")

        # Normalize to [0, 1] if needed
        if video.min() < 0:
            video = (video + 1) / 2
        video = video.clamp(0, 1)

        # Handle different 4D tensor formats -> target [F, H, W, C]
        if video.dim() == 4:
            if video.shape[0] == 3:  # [C, F, H, W] - channels first
                video = video.permute(1, 2, 3, 0)  # -> [F, H, W, C]
            elif video.shape[1] == 3:  # [F, C, H, W] - frames first, channels second
                video = video.permute(0, 2, 3, 1)  # -> [F, H, W, C]
            # else: assume already [F, H, W, C]
        
        logger.debug(f"Final video shape before numpy: {video.shape}")

        video = (video.cpu().float().numpy() * 255).astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in video]
        return frames

    def _apply_cfg_rescale(
        self,
        noise_pred_cond: torch.Tensor,
        noise_pred_guided: torch.Tensor,
        rescale_scale: float = 0.7,
    ) -> torch.Tensor:
        """Rescale guided prediction to match conditional prediction variance.
        
        This prevents over-saturation artifacts that can occur with high CFG values
        by rescaling the guided noise prediction to have similar variance as the
        conditional prediction.
        
        Based on "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/abs/2305.08891)
        
        Args:
            noise_pred_cond: Conditional noise prediction (without guidance)
            noise_pred_guided: Guided noise prediction (after CFG)
            rescale_scale: Interpolation factor between guided and rescaled prediction.
                           0.0 = fully rescaled, 1.0 = no rescaling (original guided)
        
        Returns:
            Rescaled noise prediction tensor
        """
        # Calculate standard deviation per sample (keeping batch dimension)
        std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_guided = noise_pred_guided.std(dim=list(range(1, noise_pred_guided.ndim)), keepdim=True)
        
        # Avoid division by zero
        std_guided = torch.clamp(std_guided, min=1e-8)
        
        # Rescale to match conditional variance
        noise_pred_rescaled = noise_pred_guided * (std_cond / std_guided)
        
        # Interpolate between rescaled and original guided prediction
        noise_pred_final = (
            rescale_scale * noise_pred_guided + 
            (1 - rescale_scale) * noise_pred_rescaled
        )
        
        return noise_pred_final

    def _get_restart_schedule(
        self,
        num_steps: int,
        num_segments: int = 2,
    ) -> List[int]:
        """Calculate restart points for Restart sampling.
        
        Restart sampling (arXiv:2306.14878) divides the denoising process into
        segments and adds controlled noise at restart points to improve quality.
        
        For Res_2s (2 segments), with 50 steps:
        - Segment 1: steps 0-25
        - Restart: add noise at step 25
        - Segment 2: steps 25-50
        
        Args:
            num_steps: Total number of inference steps
            num_segments: Number of segments (2 = Res_2s, 3 = Res_3s)
        
        Returns:
            List of step indices where restart (noise injection) should occur
        """
        if num_segments <= 1:
            return []  # No restarts for single segment
        
        # Calculate segment size
        segment_size = num_steps // num_segments
        
        # Restart points are at segment boundaries (excluding start and end)
        restart_points = [segment_size * (i + 1) for i in range(num_segments - 1)]
        
        logger.debug(f"Restart schedule: {num_segments} segments, restart at steps {restart_points}")
        return restart_points

    def _apply_restart_noise(
        self,
        latents: torch.Tensor,
        scheduler,
        current_step: int,
        restart_strength: float = 0.3,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply noise injection at restart points for improved motion quality.
        
        At restart points, we add controlled noise back to the latents using
        the scheduler's scale_noise method. This helps break the deterministic
        trajectory and can produce more dynamic motion.
        
        Args:
            latents: Current latent tensor
            scheduler: The flow matching scheduler
            current_step: Current step index
            restart_strength: Amount of noise to add (0.0-1.0)
            generator: Random generator for reproducibility
        
        Returns:
            Latents with restart noise applied
        """
        if restart_strength <= 0:
            return latents
            
        # Get the current sigma/timestep
        sigmas = scheduler.sigmas
        if current_step < len(sigmas):
            timestep = sigmas[current_step]
        else:
            return latents
            
        # Generate noise and scale it by restart_strength
        noise = torch.randn_like(latents, generator=generator)
        
        # Apply scaled noise using the scheduler's scale_noise method
        # This respects the flow matching formulation
        noisy_latents = scheduler.scale_noise(
            latents,
            timestep * restart_strength,  # Scale timestep by restart strength
            noise
        )
        
        logger.debug(f"Applied restart noise at step {current_step} with strength {restart_strength}")
        return noisy_latents

    def _create_restart_callback(
        self,
        restart_params: "RestartSamplerParams",
        total_steps: int,
        generator: Optional[torch.Generator] = None,
    ):
        """Create a callback function for Restart sampling during inference.
        
        The callback injects noise at restart points to improve motion quality,
        implementing the Restart sampling technique (arXiv:2306.14878).
        
        Args:
            restart_params: RestartSamplerParams configuration
            total_steps: Total number of inference steps
            generator: Random generator for reproducibility
        
        Returns:
            Callback function for use with callback_on_step_end
        """
        restart_points = self._get_restart_schedule(
            total_steps, 
            restart_params.num_segments
        )
        
        if not restart_points:
            return None
        
        logger.info(f"Restart sampling enabled: {restart_params.num_segments} segments, "
                   f"restart at steps {restart_points}, strength={restart_params.restart_strength}")
        
        def restart_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            """Callback to apply restart noise at designated steps."""
            if step in restart_points:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    # Apply restart noise
                    noisy_latents = self._apply_restart_noise(
                        latents,
                        pipe.scheduler,
                        step,
                        restart_params.restart_strength,
                        generator,
                    )
                    callback_kwargs["latents"] = noisy_latents
                    logger.debug(f"Restart noise applied at step {step}/{total_steps}")
            
            return callback_kwargs
        
        return restart_callback

    def _create_abort_callback(self):
        """Create callback to check abort flag during video generation."""
        from ..abort import abort_controller
        def callback(pipeline, step_index, timestep, callback_kwargs):
            if abort_controller.should_abort():
                raise InterruptedError("Video generation aborted by client")
            return callback_kwargs
        return callback

    def _compose_callbacks(self, *callbacks):
        """Chain multiple step-end callbacks together."""
        active = [cb for cb in callbacks if cb is not None]
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        def composed(pipe, step, timestep, callback_kwargs):
            for cb in active:
                callback_kwargs = cb(pipe, step, timestep, callback_kwargs)
            return callback_kwargs
        return composed

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: int = 41,
        steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        smart_decode: bool = True,
        loras: Optional[list[dict]] = None,
        target_resolution: int = 768,
        use_gradient_estimation: bool = False,
        ge_gamma: float = 2.0,
        guider_params: Optional[LTXGuiderParams] = None,
        img_compression: int = 42,
        add_motion_negative: bool = True,
        restart_params: Optional[RestartSamplerParams] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate video frames.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to reduce artifacts
            image: Optional source image for image-to-video mode
            width: Output width (auto-calculated if None and image provided)
            height: Output height (auto-calculated if None and image provided)
            num_frames: Number of frames to generate (must be 8n+1, e.g., 41)
            steps: Number of inference steps (default 30)
            guidance_scale: Guidance scale (default 3.0)
            seed: Random seed for reproducibility
            smart_decode: Use memory-optimized decoding (default True)
            target_resolution: Target average resolution for dimension scaling (default 768)
            use_gradient_estimation: Enable gradient estimation for faster inference (default False)
            ge_gamma: Gradient estimation coefficient (default 2.0, higher = more aggressive)
            guider_params: Optional LTXGuiderParams for enhanced guidance (CFG, STG, rescaling)
            img_compression: Compression level 0-100 for image preprocessing (default 42)
                            Higher values = more compression. Prevents "still video" output.
            add_motion_negative: Append motion-related negative prompt (default True). Helps prevent static video output.
            restart_params: Optional RestartSamplerParams for Restart sampling (Res_2s, Res_3s).
                           Enable for improved motion quality. Uses scheduler stochasticity params.
            **kwargs: Additional arguments
        
        Returns:
            List of PIL Image frames
        """
        self.touch()
        if not self.is_loaded:
            self.load()

        # Handle LoRAs
        if loras:
            lora_manager = LoRAManager()
            lora_manager.manage_lora_state(self._pipe, loras)
        elif hasattr(self, '_last_had_loras') and self._last_had_loras:
            # Clear LoRAs if previous request had them but this one doesn't
            lora_manager = LoRAManager()
            lora_manager.manage_lora_state(self._pipe, None)
        self._last_had_loras = bool(loras)

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Pre-generation cleanup
        torch.cuda.empty_cache()

        logger.info(f"Generating LTX video: prompt='{prompt[:50]}...', frames={num_frames}, steps={steps}")

        # Apply motion-related negative prompt if enabled
        effective_negative_prompt = negative_prompt
        if add_motion_negative:
            if effective_negative_prompt:
                effective_negative_prompt = f"{effective_negative_prompt}, {self.DEFAULT_MOTION_NEGATIVE}"
            else:
                effective_negative_prompt = self.DEFAULT_MOTION_NEGATIVE
            logger.debug(f"Added motion negative prompt: {self.DEFAULT_MOTION_NEGATIVE}")

        # Apply guider_params if provided - use cfg_scale for guidance
        effective_guidance_scale = guidance_scale
        stg_scale = 0.0  # Default: no spatio-temporal guidance
        stg_blocks = []
        rescale_scale = None  # Only apply rescaling if guider_params provided
        
        if guider_params is not None:
            effective_guidance_scale = guider_params.cfg_scale
            stg_scale = guider_params.stg_scale
            stg_blocks = guider_params.stg_blocks
            rescale_scale = guider_params.rescale_scale
            logger.info(f"Using guider params: cfg={effective_guidance_scale}, stg={stg_scale}, "
                       f"stg_blocks={stg_blocks}, rescale={rescale_scale}")

        # Set up Restart sampling callback if enabled
        restart_callback = None
        if restart_params is not None:
            restart_callback = self._create_restart_callback(
                restart_params, steps, generator
            )

        # Determine dimensions and mode
        if image is not None:
            # Image-to-video mode
            if self._mode != "image-to-video":
                logger.warning("Image provided but pipeline loaded in text-to-video mode. Reloading...")
                self.unload()
                self._mode = "image-to-video"
                self.load()

            # Auto-calculate dimensions using target_resolution
            # Use _calculate_scaled_dimensions to preserve aspect ratio while targeting avg resolution
            if width is None or height is None:
                gen_width, gen_height = self._calculate_scaled_dimensions(
                    image.width, image.height, target_avg=target_resolution
                )
                logger.info(f"Scaled to target resolution {target_resolution}: {gen_width}x{gen_height} (source: {image.width}x{image.height})")
            else:
                gen_width, gen_height = width, height

            # Resize image to target dimensions
            image = image.resize((gen_width, gen_height))

            # CRITICAL: Apply preprocessing for motion - prevents "still video" output
            image = self._preprocess_image_for_video(image, img_compression)
            logger.info(f"Applied LTXVPreprocess with compression={img_compression}")

            # Build pipeline kwargs for image-to-video
            # Note: image_cond_noise_scale is NOT available in LTXImageToVideoPipeline
            # decode_noise_scale handles temporal variance during VAE decode
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": effective_negative_prompt,
                "image": image,
                "width": gen_width,
                "height": gen_height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": effective_guidance_scale,
                "generator": generator,
                "decode_timestep": 0.05,
                "decode_noise_scale": 0.025,
            }
            
            # Add STG parameters only if pipeline supports them (LTX-2 feature, not in Diffusers LTX-Video v1)
            if guider_params is not None and stg_scale > 0:
                import inspect
                pipe_signature = inspect.signature(self._pipe.__call__)
                if 'stg_scale' in pipe_signature.parameters:
                    pipeline_kwargs["stg_scale"] = stg_scale
                    if stg_blocks:
                        pipeline_kwargs["stg_skip_layers"] = stg_blocks
                    logger.debug(f"STG enabled: scale={stg_scale}, blocks={stg_blocks}")
                else:
                    logger.warning("STG parameters requested but not supported by this pipeline version (Diffusers LTX-Video)")

            # Always include abort callback; compose with restart if needed
            abort_callback = self._create_abort_callback()
            step_callback = self._compose_callbacks(abort_callback, restart_callback)
            if step_callback is not None:
                pipeline_kwargs["callback_on_step_end"] = step_callback
                pipeline_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            if smart_decode:
                output = self._pipe(**pipeline_kwargs, output_type="latent")
                latents = output.frames
                frames = self._smart_decode(
                    latents, num_frames, gen_height, gen_width,
                    decode_timestep=0.05, decode_noise_scale=0.025
                )
            else:
                output = self._pipe(**pipeline_kwargs)
                frames = output.frames[0]

        else:
            # Text-to-video mode
            if self._mode != "text-to-video":
                logger.warning("No image provided but pipeline loaded in image-to-video mode. Reloading...")
                self.unload()
                self._mode = "text-to-video"
                self.load()

            # Default to landscape using target_resolution (3:2 aspect ratio)
            # Calculate dimensions to achieve target average resolution
            if width is None and height is None:
                gen_width, gen_height = self._calculate_scaled_dimensions(
                    768, 512, target_avg=target_resolution  # Use 3:2 landscape as base aspect
                )
            else:
                gen_width = width if width is not None else 768
                gen_height = height if height is not None else 512

            # Build pipeline kwargs with correct parameter names
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": effective_negative_prompt,
                "width": gen_width,
                "height": gen_height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": effective_guidance_scale,
                "generator": generator,
                "decode_timestep": 0.05,  # Official LTX recommended value
            }
            
            # Add STG parameters only if pipeline supports them (LTX-2 feature, not in Diffusers LTX-Video v1)
            if guider_params is not None and stg_scale > 0:
                import inspect
                pipe_signature = inspect.signature(self._pipe.__call__)
                if 'stg_scale' in pipe_signature.parameters:
                    pipeline_kwargs["stg_scale"] = stg_scale
                    if stg_blocks:
                        pipeline_kwargs["stg_skip_layers"] = stg_blocks
                    logger.debug(f"STG enabled: scale={stg_scale}, blocks={stg_blocks}")
                else:
                    logger.warning("STG parameters requested but not supported by this pipeline version (Diffusers LTX-Video)")

            # Always include abort callback; compose with restart if needed
            abort_callback = self._create_abort_callback()
            step_callback = self._compose_callbacks(abort_callback, restart_callback)
            if step_callback is not None:
                pipeline_kwargs["callback_on_step_end"] = step_callback
                pipeline_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            if smart_decode:
                output = self._pipe(**pipeline_kwargs, output_type="latent")
                latents = output.frames
                frames = self._smart_decode(
                    latents, num_frames, gen_height, gen_width,
                    decode_timestep=0.05
                )
            else:
                output = self._pipe(**pipeline_kwargs)
                frames = output.frames[0]

        # Post-generation cleanup
        torch.cuda.empty_cache()

        logger.info(f"Generated {len(frames)} frames")
        return frames

    def fast_generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: int = 41,
        steps: int = 25,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        smart_decode: bool = True,
        loras: Optional[list[dict]] = None,
        target_resolution: int = 768,
        ge_gamma: float = 2.0,
        guider_params: Optional["LTXGuiderParams"] = None,
        add_motion_negative: bool = True,
        restart_params: Optional["RestartSamplerParams"] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate video frames using gradient estimation for faster inference.
        
        Convenience method that enables gradient estimation with optimized defaults
        for batch processing. Uses 25 steps (vs 50) while maintaining quality.
        
        Particularly effective for batch processing multiple videos where the
        reduced inference time compounds.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to reduce artifacts
            image: Optional source image for image-to-video mode
            width: Output width (auto-calculated if None and image provided)
            height: Output height (auto-calculated if None and image provided)
            num_frames: Number of frames to generate (must be 8n+1, e.g., 41)
            steps: Number of inference steps (default 25 with gradient estimation)
            guidance_scale: Guidance scale (default 5.0)
            seed: Random seed for reproducibility
            smart_decode: Use memory-optimized decoding (default True)
            loras: List of LoRA configurations
            target_resolution: Target average resolution for dimension scaling (default 768)
            ge_gamma: Gradient estimation coefficient (default 2.0)
            guider_params: Optional LTXGuiderParams for enhanced guidance
            add_motion_negative: Append motion-related negative prompt (default True). Helps prevent static video output.
            restart_params: Optional RestartSamplerParams for Restart sampling (Res_2s, Res_3s).
            **kwargs: Additional arguments
        
        Returns:
            List of PIL Image frames
        """
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            smart_decode=smart_decode,
            loras=loras,
            target_resolution=target_resolution,
            use_gradient_estimation=True,
            ge_gamma=ge_gamma,
            guider_params=guider_params,
            add_motion_negative=add_motion_negative,
            restart_params=restart_params,
            **kwargs,
        )
