"""Corruption recovery wrapper for image generation."""
import logging, random
from functools import wraps
from PIL import Image
from .validation import check_image_quality
from .vram.manager import vram_manager
from .abort import abort_controller, GenerationAbortedError

logger = logging.getLogger(__name__)


def is_oom_error(error: Exception) -> bool:
    """Check if error is CUDA out of memory."""
    error_str = str(error).lower()
    return "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str)


def safe_generate(pipeline, generate_kwargs: dict, max_retries: int = 2, max_oom_retries: int = 2) -> Image.Image:
    """Wrapper that validates output and retries with cache clearing/seed changes."""
    last_error = None
    oom_retry_count = 0
    
    for attempt in range(max_retries + 1):
        # Check if generation was aborted before each attempt
        if abort_controller.should_abort():
            raise GenerationAbortedError("Generation aborted")
        
        try:
            image = pipeline.generate(**generate_kwargs)
            is_valid, reason = check_image_quality(image)
            if is_valid:
                if attempt > 0:
                    logger.info(f"Generation succeeded on attempt {attempt + 1}")
                return image
            last_error = reason
            logger.warning(f"Attempt {attempt + 1}: validation failed - {reason}")
            # Black image detection triggers reload
            if "black" in reason.lower() or "blank" in reason.lower():
                logger.warning("Black/blank image detected, triggering pipeline reload")
                vram_manager.nuclear_cleanup()
                if hasattr(pipeline, "unload"): pipeline.unload()
                if hasattr(pipeline, "load"): pipeline.load()
            elif attempt < max_retries:
                vram_manager.cleanup()
                if "seed" in generate_kwargs:
                    generate_kwargs["seed"] = random.randint(0, 2**32 - 1)
        except (InterruptedError, GenerationAbortedError):
            raise  # Don't retry abort/disconnect — propagate immediately
        except Exception as e:
            last_error = str(e)
            logger.error(f"Attempt {attempt + 1}: generation error - {e}")
            
            # OOM-specific handling
            if is_oom_error(e) and oom_retry_count < max_oom_retries:
                oom_retry_count += 1
                logger.warning(f"OOM error detected, attempting recovery (OOM retry {oom_retry_count}/{max_oom_retries})")
                vram_manager.nuclear_cleanup()
                if hasattr(pipeline, "unload"): pipeline.unload()
                if hasattr(pipeline, "load"): pipeline.load()
                logger.info(f"Pipeline reloaded after OOM, retrying generation")
                continue
            elif attempt < max_retries:
                vram_manager.cleanup()
                
    # Final attempt: reload pipeline
    if abort_controller.should_abort():
        raise GenerationAbortedError("Generation aborted")
    logger.warning("All retries exhausted, attempting pipeline reload")
    try:
        if hasattr(pipeline, "unload"): pipeline.unload()
        if hasattr(pipeline, "load"): pipeline.load()
        image = pipeline.generate(**generate_kwargs)
        is_valid, reason = check_image_quality(image)
        if is_valid:
            logger.info("Generation succeeded after pipeline reload")
            return image
        last_error = reason
    except (InterruptedError, GenerationAbortedError):
        raise
    except Exception as e:
        last_error = str(e)
    raise RuntimeError(f"Generation failed after {max_retries + 2} attempts: {last_error}")


def validated_generation(func):
    """Decorator that wraps any generate function with validation and retry logic."""
    @wraps(func)
    def wrapper(*args, max_retries: int = 2, **kwargs):
        last_error = None
        for attempt in range(max_retries + 1):
            if abort_controller.should_abort():
                raise GenerationAbortedError("Generation aborted")
            try:
                image = func(*args, **kwargs)
                is_valid, reason = check_image_quality(image)
                if is_valid:
                    return image
                last_error = reason
                logger.warning(f"Validation failed on attempt {attempt + 1}: {reason}")
                vram_manager.cleanup()
                if "seed" in kwargs:
                    kwargs["seed"] = random.randint(0, 2**32 - 1)
            except (InterruptedError, GenerationAbortedError):
                raise
            except Exception as e:
                last_error = str(e)
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                vram_manager.cleanup()
        raise RuntimeError(f"Generation failed after {max_retries + 1} attempts: {last_error}")
    return wrapper


def robust_generate(pipeline, generate_kwargs: dict, max_oom_retries: int = 2) -> Image.Image:
    """Generation with OOM recovery and black image detection.
    
    This wrapper provides comprehensive error handling:
    - Detects and recovers from CUDA OOM errors with nuclear cleanup
    - Detects black/blank images and triggers pipeline reload
    - Tracks OOM retry count separately from general retries
    """
    last_error = None
    oom_retry_count = 0
    max_general_retries = 2
    
    attempt = 0
    while attempt <= max_general_retries:
        # Check if generation was aborted before each attempt
        if abort_controller.should_abort():
            raise GenerationAbortedError("Generation aborted")
        
        try:
            logger.debug(f"robust_generate: attempt {attempt + 1}")
            image = pipeline.generate(**generate_kwargs)
            
            # Validate output
            is_valid, reason = check_image_quality(image)
            if is_valid:
                if attempt > 0 or oom_retry_count > 0:
                    logger.info(f"Generation succeeded (attempt {attempt + 1}, OOM retries: {oom_retry_count})")
                return image
            
            last_error = reason
            logger.warning(f"Attempt {attempt + 1}: validation failed - {reason}")
            
            # Black image detection triggers aggressive recovery
            if "black" in reason.lower() or "blank" in reason.lower():
                logger.warning("Black/blank image detected, performing nuclear cleanup and reload")
                vram_manager.nuclear_cleanup()
                if hasattr(pipeline, "unload"): pipeline.unload()
                if hasattr(pipeline, "load"): pipeline.load()
            else:
                vram_manager.cleanup()
                if "seed" in generate_kwargs:
                    generate_kwargs["seed"] = random.randint(0, 2**32 - 1)
            
            attempt += 1
            
        except (InterruptedError, GenerationAbortedError):
            raise  # Don't retry abort/disconnect — propagate immediately
        except Exception as e:
            last_error = str(e)
            logger.error(f"Attempt {attempt + 1}: generation error - {e}")
            
            # OOM-specific recovery
            if is_oom_error(e):
                if oom_retry_count < max_oom_retries:
                    oom_retry_count += 1
                    logger.warning(f"OOM detected, nuclear cleanup + reload (OOM retry {oom_retry_count}/{max_oom_retries})")
                    vram_manager.nuclear_cleanup()
                    if hasattr(pipeline, "unload"): pipeline.unload()
                    if hasattr(pipeline, "load"): pipeline.load()
                    # Don't increment attempt for OOM retries
                    continue
                else:
                    logger.error(f"Max OOM retries ({max_oom_retries}) exceeded")
                    raise RuntimeError(f"Generation failed: OOM retries exhausted - {last_error}")
            else:
                vram_manager.cleanup()
                attempt += 1
    
    # Final fallback: one more try after full reload
    if abort_controller.should_abort():
        raise GenerationAbortedError("Generation aborted")
    logger.warning("All retries exhausted, final attempt with pipeline reload")
    try:
        vram_manager.nuclear_cleanup()
        if hasattr(pipeline, "unload"): pipeline.unload()
        if hasattr(pipeline, "load"): pipeline.load()
        image = pipeline.generate(**generate_kwargs)
        is_valid, reason = check_image_quality(image)
        if is_valid:
            logger.info("Generation succeeded after final reload")
            return image
        last_error = reason
    except (InterruptedError, GenerationAbortedError):
        raise
    except Exception as e:
        last_error = str(e)
        if is_oom_error(e):
            raise RuntimeError(f"Generation failed: persistent OOM error - {last_error}")
    
    raise RuntimeError(f"Generation failed after all attempts: {last_error}")
