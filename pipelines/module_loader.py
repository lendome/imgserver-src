"""Module auto-discovery for pipeline modules.

This module provides automatic discovery and loading of pipeline modules
from the pipelines directory. Modules are loaded dynamically and self-register
via the @pipeline_module decorator.
"""

import logging
import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict

from .base import get_all_modules, get_module_info, ModuleInfo

logger = logging.getLogger(__name__)

# Flag to prevent re-discovery of modules
_MODULES_LOADED = False


def discover_pipeline_modules() -> List[str]:
    """
    Discover and import all pipeline modules from the pipelines directory.
    
    This function:
    1. Scans the pipelines directory for .py files
    2. Dynamically imports each module (except __init__.py, base.py, registry.py, module_loader.py)
    3. Pipeline classes self-register via the @pipeline_module decorator during import
    4. Returns list of discovered module names
    
    Returns:
        List of discovered module names (without .py extension)
    """
    discovered_modules = []
    
    # Get the directory of this file
    pipelines_dir = Path(__file__).parent
    
    # Files to skip during discovery
    skip_files = {'__init__.py', 'base.py', 'registry.py', 'module_loader.py'}
    
    logger.info(f"Starting module discovery in {pipelines_dir}")
    
    # Iterate through all .py files in the pipelines directory
    for item in pkgutil.iter_modules([str(pipelines_dir)]):
        module_name = item.name
        
        # Skip excluded files
        if f"{module_name}.py" in skip_files:
            logger.debug(f"Skipping {module_name}.py (excluded)")
            continue
        
        try:
            # Dynamically import the module using the correct package path
            # Since we're in src_new.pipelines, we need to use the relative import
            full_module_name = f"src_new.pipelines.{module_name}"
            logger.debug(f"Importing module: {full_module_name}")
            importlib.import_module(full_module_name)
            discovered_modules.append(module_name)
            logger.info(f"Successfully discovered and imported: {module_name}")
        except Exception as e:
            logger.error(f"Failed to import module {module_name}: {e}", exc_info=True)
            continue
    
    logger.info(f"Module discovery complete. Found {len(discovered_modules)} modules: {discovered_modules}")
    return discovered_modules


def ensure_modules_loaded() -> Dict[str, ModuleInfo]:
    """
    Ensure all pipeline modules are loaded and registered.
    
    This function:
    1. Calls discover_pipeline_modules() only once (uses module-level flag)
    2. Prevents re-discovery on subsequent calls
    3. Returns the module registry after loading
    
    Returns:
        Dictionary mapping module names to their ModuleInfo objects
    """
    global _MODULES_LOADED
    
    if _MODULES_LOADED:
        logger.debug("Modules already loaded, skipping re-discovery")
        return get_all_modules()
    
    logger.info("First time module loading, discovering pipeline modules...")
    discover_pipeline_modules()
    _MODULES_LOADED = True
    
    # Return the populated registry
    modules = get_all_modules()
    logger.info(f"Module loading complete. Registered modules: {list(modules.keys())}")
    
    return modules
