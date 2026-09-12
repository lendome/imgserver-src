"""Wan 2.1 Video Pipeline implementation for video generation."""

import gc
import logging
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

from .base import BasePipeline, pipeline_module

logger = logging.getLogger(__name__)


@pipeline_module(
    name="wan",
    display_name="Wan 2.1 Video Generation",
    output_type="video",
    conflicts_with=["sdxl", "zimg", "tts", "ltx"],
    checkpoint_patterns=[r"wan", r"Wan"],
    vram_estimate_gb=12.0,
    supports_checkpoints=False,
)
class WanPipeline(BasePipeline):
    """Wan 2.1 Video pipeline with aggressive VRAM management.
    
    Supports both text-to-video and image-to-video generation modes.
    Uses sequential CPU offloading to fit 14B models on 24GB VRAM.
    """

    # Model IDs for HuggingFace
    T2V_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    I2V_MODEL_ID_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    I2V_MODEL_ID_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    def __init__(self, mode: str = "text-to-video", resolution: str = "480p"):
        """Initialize Wan pipeline.
        
        Args:
            mode: Generation mode - "text-to-video" or "image-to-video"
            resolution: Target resolution - "480p" or "720p"
        """
        self._pipe = None
        self._parked = False
        self._mode = mode
        self._resolution = resolution
        self._device = "cpu"  # With CPU offload, device tracking is different

    @property
    def is_loaded(self) -> bool:
        # With CPU offload, pipeline stays on CPU but is "loaded"
        return self._pipe is not None

    @property
    def is_parked(self) -> bool:
        return False  # CPU offload doesn't support parking

    def _get_model_id(self) -> str:
        """Get the appropriate model ID based on mode and resolution."""
        if self._mode == "text-to-video":
            return self.T2V_MODEL_ID
        else:
            if self._resolution == "720p":
                return self.I2V_MODEL_ID_720P
            return self.I2V_MODEL_ID_480P

    def load(self) -> None:
        """Load Wan pipeline with CPU offloading for low VRAM usage."""
        if self.is_loaded:
            logger.debug("Wan already loaded, skipping")
            return

        model_id = self._get_model_id()
        logger.info(f"Loading Wan 2.1 pipeline with CPU offload (mode={self._mode}, resolution={self._resolution})")

        # Clear memory before loading
        gc.collect()
        torch.cuda.empty_cache()

        from diffusers import AutoencoderKLWan
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        # Configure scheduler with flow matching
        flow_shift = 5.0 if self._resolution == "720p" else 3.0
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift,
        )

        if self._mode == "image-to-video":
            from diffusers import WanImageToVideoPipeline
            from transformers import CLIPVisionModel

            # Load VAE in float32 - keep on CPU, will be offloaded
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            # Load image encoder in float32 - keep on CPU
            image_encoder = CLIPVisionModel.from_pretrained(
                model_id,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
            )

            # Load pipeline - DO NOT move to CUDA, use CPU offload instead
            self._pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
            )
        else:
            from diffusers import WanPipeline as DiffusersWanPipeline

            # Load VAE in float32 - keep on CPU
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            # Load pipeline - DO NOT move to CUDA
            self._pipe = DiffusersWanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.bfloat16,
            )

        self._pipe.scheduler = scheduler
        
        # CRITICAL: Enable sequential CPU offload BEFORE any .to("cuda") call
        # This moves each model component to GPU only when needed, then back to CPU
        # This is the key to fitting 14B models on 24GB VRAM
        self._pipe.enable_sequential_cpu_offload()
        logger.info("Wan sequential CPU offload enabled - 14B model will fit in 24GB VRAM")
        
        # Enable VAE tiling for lower memory during decode
        self._pipe.vae.enable_tiling()
        logger.debug("Wan VAE tiling enabled")
        
        self._device = "cuda"  # Logically on CUDA via offload
        self._parked = False

        logger.info(f"Wan 2.1 pipeline loaded (mode={self._mode})")

    def unload(self) -> None:
        """Unload pipeline and free VRAM."""
        if self._pipe is None:
            return

        logger.info("Unloading Wan 2.1 pipeline")

        # With sequential CPU offload, components are already mostly on CPU
        # Just delete the pipeline
        del self._pipe
        self._pipe = None
        self._device = "cpu"
        self._parked = False

        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("Wan 2.1 unloaded")

    def to_cpu(self) -> None:
        """Park pipeline to CPU - no-op with sequential offload."""
        # Sequential CPU offload already keeps model on CPU
        pass

    def to_gpu(self) -> None:
        """Restore pipeline to GPU - no-op with sequential offload."""
        # Sequential CPU offload handles GPU dynamically
        pass

    def estimate_vram(self) -> int:
        """Estimated peak VRAM usage with sequential CPU offload.
        
        With sequential offload, only one model component is on GPU at a time.
        Peak usage is typically ~10-12GB for 14B models, ~6-8GB for 1.3B.
        """
        if self._mode == "text-to-video":
            return 8 * 1024 ** 3  # 1.3B model with offload
        return 12 * 1024 ** 3  # 14B model with sequential offload

    def _calculate_dimensions(
        self,
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_area: Optional[int] = None,
    ) -> tuple:
        """Calculate output dimensions respecting Wan's requirements.
        
        Args:
            image: Source image for I2V (determines aspect ratio)
            width: Explicit width override
            height: Explicit height override
            max_area: Maximum pixel area (default based on resolution)
            
        Returns:
            Tuple of (width, height)
        """
        if width is not None and height is not None:
            return width, height

        # Default max areas by resolution
        if max_area is None:
            max_area = 720 * 1280 if self._resolution == "720p" else 480 * 832

        if image is not None:
            # Calculate from image aspect ratio
            aspect_ratio = image.height / image.width
            mod_value = self._pipe.vae_scale_factor_spatial * self._pipe.transformer.config.patch_size[1]
            calc_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            calc_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            return calc_width, calc_height

        # Default dimensions for T2V
        if self._resolution == "720p":
            return 1280, 720
        return 832, 480

    def _create_abort_callback(self):
        """Create callback to check abort flag during video generation."""
        from ..abort import abort_controller
        def callback(pipeline, step_index, timestep, callback_kwargs):
            if abort_controller.should_abort():
                raise InterruptedError("Video generation aborted by client")
            return callback_kwargs
        return callback

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: int = 81,
        steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate video frames.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to reduce artifacts
            image: Optional source image for image-to-video mode
            width: Output width (auto-calculated if None)
            height: Output height (auto-calculated if None)
            num_frames: Number of frames to generate (default 81)
            steps: Number of inference steps (default 50)
            guidance_scale: Guidance scale (default 5.0)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
            
        Returns:
            List of PIL Image frames
        """
        self.touch()

        # Switch mode if needed based on image presence
        if image is not None and self._mode != "image-to-video":
            logger.warning("Image provided but pipeline loaded in T2V mode. Reloading for I2V...")
            self.unload()
            self._mode = "image-to-video"
            self.load()
        elif image is None and self._mode == "image-to-video":
            logger.warning("No image provided but pipeline loaded in I2V mode. Reloading for T2V...")
            self.unload()
            self._mode = "text-to-video"
            self.load()

        if not self.is_loaded:
            self.load()

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Pre-generation cleanup
        torch.cuda.empty_cache()

        # Calculate dimensions
        gen_width, gen_height = self._calculate_dimensions(image, width, height)

        logger.info(f"Generating Wan video: prompt='{prompt[:50]}...', frames={num_frames}, "
                   f"size={gen_width}x{gen_height}, steps={steps}")

        # Default negative prompt for Wan
        if not negative_prompt:
            negative_prompt = (
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "worst quality, low quality, JPEG compression residue, ugly, "
                "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
                "messy background, three legs, walking backwards"
            )

        # Build pipeline kwargs
        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": gen_height,
            "width": gen_width,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Add image for I2V mode
        if image is not None:
            # Resize image to target dimensions
            image = image.resize((gen_width, gen_height))
            pipeline_kwargs["image"] = image

        # Generate with abort callback
        pipeline_kwargs["callback_on_step_end"] = self._create_abort_callback()
        output = self._pipe(**pipeline_kwargs)
        frames = output.frames[0]

        # Post-generation cleanup
        torch.cuda.empty_cache()

        logger.info(f"Generated {len(frames)} frames")
        return frames
