"""Route blueprints for the image server."""

from .sdxl import sdxl_bp
from .zimg import zimg_bp
from .generate import generate_bp
from .checkpoint import checkpoint_bp
from .cancel import cancel_bp
from .classify import classify_bp, classify_root_bp
from .tts import tts_bp
from .ltx import ltx_bp
from .wan import wan_bp


def get_all_blueprints():
    """Get all registered route blueprints."""
    return [
        sdxl_bp,
        zimg_bp,
        generate_bp,
        checkpoint_bp,
        cancel_bp,
        classify_bp,
        classify_root_bp,
        tts_bp,
        ltx_bp,
        wan_bp,
    ]


__all__ = [
    "sdxl_bp",
    "zimg_bp",
    "generate_bp",
    "checkpoint_bp",
    "cancel_bp",
    "classify_bp",
    "classify_root_bp",
    "tts_bp",
    "ltx_bp",
    "wan_bp",
    "get_all_blueprints",
]
