"""Z-Image Turbo pipeline implementation with VRAM optimizations."""

import copy
import gc
import logging
import os
import time
from typing import Optional

import torch
from PIL import Image
from diffusers import ZImagePipeline as DiffusersZImagePipeline

from .base import BasePipeline, pipeline_module
from ..config import get_config
from ..vram.manager import vram_manager
from ..vram.ram_manager import ram_manager
from ..abort import abort_controller

logger = logging.getLogger(__name__)

# Module-level cache for shared components (text encoder, tokenizer, VAE, scheduler)
# These are expensive to load but identical across all Z-Image checkpoints
_cached_components = None

# TF32 optimizations for better performance and reduced VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


@pipeline_module(
    name="zimg",
    display_name="Z-Image Turbo",
    output_type="image",
    conflicts_with=["sdxl", "tts", "ltx"],  # Must unload these before loading zimg
    checkpoint_patterns=[r"zimage", r"z-image", r"z_image", r"zimg", r"z-turbo", r"zturbo", r"moodyv", r"moody.*v\d+", r"realdream"],
    vram_estimate_gb=12.0,
    supports_checkpoints=True
)
class ZImagePipeline(BasePipeline):
    """Z-Image Turbo pipeline with optimized VRAM management."""
    
    def __init__(self, checkpoint: Optional[str] = None):
        """Initialize Z-Image pipeline.
        
        Args:
            checkpoint: Optional local checkpoint filename. If provided, loads from
                       local file. Otherwise loads from HuggingFace model ID.
        """
        self._pipe = None
        self._parked = False
        self._checkpoint = checkpoint
        self._config = get_config()
        
        # Resolve checkpoint path if local file
        if checkpoint:
            from pathlib import Path
            checkpoint_dir = Path(self._config.sdxl_checkpoints_dir)
            self.checkpoint_path = str(checkpoint_dir / checkpoint)
        else:
            self.checkpoint_path = None
    
    @property
    def is_loaded(self) -> bool:
        return (
            self._pipe is not None 
            and hasattr(self._pipe, 'device') 
            and self._pipe.device.type == 'cuda'
        )

    @property
    def is_parked(self) -> bool:
        return self._parked and self._pipe is not None
    
    def _get_cached_components(self):
        """Get or load shared components (text encoder, tokenizer, VAE, scheduler).
        
        These components are identical across all Z-Image checkpoints, so we
        cache them module-wide for fast checkpoint switching.
        """
        global _cached_components
        
        if _cached_components is not None:
            logger.debug("Using cached Z-Image components")
            return _cached_components
        
        logger.info("Loading Z-Image base components (one-time cost)...")
        t0 = time.time()
        
        config = self._config
        
        # Load full base pipeline to extract components
        base_pipe = DiffusersZImagePipeline.from_pretrained(
            config.zimage_model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        # Extract and cache components
        _cached_components = {
            "text_encoder": base_pipe.text_encoder,
            "tokenizer": base_pipe.tokenizer,
            "vae": base_pipe.vae,
            "scheduler": base_pipe.scheduler,
        }
        
        # Clear the base pipeline but keep components
        del base_pipe.transformer
        del base_pipe
        gc.collect()
        
        logger.info(f"Z-Image base components loaded in {time.time() - t0:.1f}s")
        return _cached_components
    
    def load(self) -> None:
        """Load Z-Image model with optimizations.
        
        OPTIMIZATION: When loading local checkpoints, we:
        1. Load/cache shared components (text_encoder, tokenizer, VAE) ONCE
        2. Load ONLY the transformer from the checkpoint using from_single_file
        3. Build pipeline manually from components
        
        This avoids downloading the full base model for every checkpoint load.
        """
        if self.is_loaded:
            return
        
        start_time = time.time()
        
        # Clear VRAM before loading
        vram_manager.cleanup()
        
        needed_ram = self.estimate_vram()
        if not ram_manager.can_park_model(needed_ram):
            logger.warning(f"Low RAM for Z-Image: need {needed_ram/1e9:.1f}GB")

        config = self._config
        
        # Load from local checkpoint or HuggingFace
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            logger.info(f"Loading Z-Image with local checkpoint: {self.checkpoint_path}")
            
            # FAST PATH: Load only transformer from checkpoint, reuse cached components
            # This is MUCH faster than from_pretrained which downloads everything
            
            # Step 1: Get cached shared components (one-time cost)
            t1 = time.time()
            components = self._get_cached_components()
            logger.info(f"Components ready in {time.time() - t1:.1f}s")
            
            # Step 2: Load transformer directly from checkpoint using from_single_file
            t2 = time.time()
            try:
                from diffusers import ZImageTransformer2DModel
                
                logger.info("Loading transformer from checkpoint (fast path)...")
                transformer = ZImageTransformer2DModel.from_single_file(
                    self.checkpoint_path,
                    config=config.zimage_model_id,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                logger.info(f"Transformer loaded in {time.time() - t2:.1f}s")
                
            except Exception as e:
                logger.warning(f"from_single_file failed ({e}), falling back to weight conversion")
                # Fallback: manual weight conversion
                transformer = self._load_transformer_fallback(components)
            
            # Step 3: Build pipeline from components (instant)
            t3 = time.time()
            self._pipe = DiffusersZImagePipeline(
                text_encoder=components["text_encoder"],
                tokenizer=components["tokenizer"],
                vae=components["vae"],
                transformer=transformer,
                scheduler=copy.deepcopy(components["scheduler"]),
            )
            logger.info(f"Pipeline built in {time.time() - t3:.2f}s")
            
        else:
            # No local checkpoint - load from HuggingFace
            logger.info(f"Loading Z-Image from {config.zimage_model_id}")
            t1 = time.time()
            self._pipe = DiffusersZImagePipeline.from_pretrained(
                config.zimage_model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            logger.info(f"Loaded from HuggingFace in {time.time() - t1:.1f}s")
        
        # Move to GPU
        t_cuda = time.time()
        free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
        if free_vram < 10:
            logger.info(f"Low VRAM ({free_vram:.1f}GB) - enabling CPU offload")
            self._pipe.enable_sequential_cpu_offload()
        else:
            self._pipe.to("cuda")
        logger.info(f"Moved to CUDA in {time.time() - t_cuda:.1f}s")
        
        self._parked = False
        
        # Enable memory optimizations
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            logger.debug("Z-Image xformers enabled")
        except Exception:
            logger.debug("Z-Image xformers not available")
        
        # Enable VAE slicing (NOT tiling - causes grid artifacts)
        if hasattr(self._pipe, 'vae'):
            try:
                self._pipe.vae.enable_slicing()
                logger.debug("Z-Image VAE slicing enabled")
            except Exception:
                pass
        
        vram_manager.cleanup()
        
        total_time = time.time() - start_time
        logger.info(f"Z-Image loaded with optimizations in {total_time:.1f}s")
        
        # Apply torch.compile for faster inference
        self._compile_for_speed()
    
    def _load_transformer_fallback(self, components):
        """Fallback: Load transformer via weight conversion if from_single_file fails."""
        from safetensors import safe_open
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers
        
        logger.info("Loading transformer via weight conversion (fallback)...")
        
        # Load the raw checkpoint state dict
        original_state_dict = {}
        with safe_open(self.checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_state_dict[key] = f.get_tensor(key)
        
        # Convert to diffusers format
        converted_state_dict = convert_z_image_transformer_checkpoint_to_diffusers(
            original_state_dict
        )
        
        # Strip prefix for diffusers compatibility
        final_state_dict = {}
        for k, v in converted_state_dict.items():
            new_key = k.replace("model.diffusion_model.", "")
            if not new_key.startswith("vae.") and not new_key.startswith("text_encoder."):
                final_state_dict[new_key] = v
        
        # Create transformer from config and load weights
        from diffusers import ZImageTransformer2DModel
        transformer = ZImageTransformer2DModel.from_pretrained(
            self._config.zimage_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        missing, unexpected = transformer.load_state_dict(final_state_dict, strict=False)
        if missing:
            logger.debug(f"Missing transformer keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Unexpected transformer keys: {len(unexpected)}")
        
        logger.info(f"Loaded {len(final_state_dict)} transformer weights via fallback")
        return transformer
    
    def unload(self) -> None:
        """Unload pipeline and free VRAM."""
        if self._pipe is not None:
            try:
                self._pipe.to("cpu")
            except Exception:
                pass
            del self._pipe
            self._pipe = None
            self._parked = False
            
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Z-Image unloaded")

    def to_cpu(self) -> None:
        """Park pipeline to CPU."""
        if self._pipe is None or self._parked:
            return
        
        logger.info("Parking Z-Image to CPU")
        self._pipe.to("cpu")
        self._parked = True
        
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Z-Image parked")

    def to_gpu(self) -> None:
        """Restore pipeline to GPU."""
        if self._pipe is None or not self._parked:
            return
        
        logger.info("Restoring Z-Image to GPU")
        self._pipe.to("cuda")
        self._parked = False
        logger.info("Z-Image restored")
    
    def _compile_for_speed(self) -> None:
        """Apply runtime optimizations for faster inference.
        
        NOTE: torch.compile requires Triton which isn't available on Windows.
        Instead, we ensure all other optimizations are properly enabled.
        """
        if self._pipe is None:
            return
        
        config = get_config()
        if not getattr(config, 'tensorrt_enabled', False):
            return
        
        logger.info("Ensuring runtime optimizations for Z-Image...")
        
        try:
            import torch
            
            # Ensure TF32 is enabled
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            logger.info("Z-Image optimizations enabled (TF32, cuDNN benchmark)")
            logger.info("NOTE: First generation will be slower due to JIT warmup")
            
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
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
    
    def _decode_latents_safe(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents with proper dtype handling and OOM fallback.
        
        Args:
            latents: Latent tensor from pipeline
            
        Returns:
            Decoded PIL Image
        """
        vae = self._pipe.vae
        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor()
        
        # Get VAE dtype (should be bfloat16)
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        
        # Scale latents by VAE scaling factor
        scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
        latents = latents / scaling_factor
        
        # Ensure latents match VAE dtype (fixes float32 vs bfloat16 mismatch)
        latents = latents.to(dtype=vae_dtype, device=vae_device)
        
        try:
            # Decode with autocast for bfloat16 stability
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    image = vae.decode(latents, return_dict=False)[0]
            return processor.postprocess(image, output_type="pil")[0]
        except torch.cuda.OutOfMemoryError:
            logger.warning("VAE decode OOM - falling back to CPU")
            torch.cuda.empty_cache()
            
            # Move VAE to CPU for decode
            vae.to("cpu")
            latents_cpu = latents.to("cpu", dtype=torch.float32)  # CPU works better with float32
            
            with torch.no_grad():
                image = vae.decode(latents_cpu, return_dict=False)[0]
            
            # Postprocess and restore VAE
            result = processor.postprocess(image, output_type="pil")[0]
            vae.to(vae_device)
            return result

    
    def generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 6,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image with optimized VRAM usage."""
        self.touch()
        
        if not self.is_loaded:
            self.load()
        
        config = get_config()
        width = width or config.default_width
        height = height or config.default_height
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Pre-generation cleanup
        torch.cuda.empty_cache()
        
        # Generate with bfloat16 autocast (Z-Image is trained for bfloat16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.inference_mode():
                result = self._pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=generator,
                    output_type="latent",  # Get latents for safe decode
                    callback_on_step_end=self._create_abort_callback(),
                )
        
        # Post-generation cleanup
        torch.cuda.empty_cache()
        
        # Decode latents with proper dtype handling
        latents = result.images
        return self._decode_latents_safe(latents)

    
    def estimate_vram(self) -> int:
        """Estimated VRAM usage (~12GB with optimizations)."""
        return 12 * 1024 ** 3

