"""
src_new - Reimagined image generation server.

A simple, self-maintaining server supporting SDXL and Z-Image pipelines.
"""

from .pipelines import get_pipeline, unload_pipeline, unload_all

__all__ = ["get_pipeline", "unload_pipeline", "unload_all"]
