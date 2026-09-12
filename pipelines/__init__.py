"""Pipeline registry with lazy loading and LRU tracking."""

from .base import BasePipeline, pipeline_module, get_module_info, get_all_modules, ModuleInfo
from .tts import TTSPipeline
from .ltx import LTXPipeline, LTXGuiderParams, RestartSamplerParams
from .wan import WanPipeline
from .registry import (
    get_pipeline,
    unload_pipeline,
    unload_all,
    get_loaded_pipelines,
    get_active_checkpoint,
)

__all__ = [
    "BasePipeline",
    "TTSPipeline",
    "LTXPipeline",
    "LTXGuiderParams",
    "RestartSamplerParams",
    "WanPipeline",
    "pipeline_module",
    "get_module_info",
    "get_all_modules",
    "ModuleInfo",
    "get_pipeline",
    "unload_pipeline",
    "unload_all",
    "get_loaded_pipelines",
    "get_active_checkpoint",
]
