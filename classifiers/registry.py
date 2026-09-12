"""Classifier registry with hot-swapping support."""

import gc
import threading
import logging
from typing import Optional, Dict, Type

from .base import BaseClassifier

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()


class ClassifierRegistry:
    """
    Manages CPU classifiers with hot-swapping.
    
    Only one classifier can be loaded at a time to conserve RAM.
    Loading a new classifier automatically unloads the current one.
    """
    
    def __init__(self):
        self._classifiers: Dict[str, Type[BaseClassifier]] = {}
        self._current: Optional[BaseClassifier] = None
        self._current_name: Optional[str] = None
    
    def register(self, name: str, classifier_cls: Type[BaseClassifier]) -> None:
        """Register a classifier class."""
        self._classifiers[name] = classifier_cls
        logger.debug(f"Registered classifier: {name}")
    
    def get(self, name: str) -> BaseClassifier:
        """
        Get a classifier by name, loading it if needed.
        
        If a different classifier is currently loaded, it will be
        unloaded first (hot-swap).
        """
        with _registry_lock:
            # Already loaded?
            if self._current_name == name and self._current is not None:
                return self._current
            
            # Need to swap - unload current first
            if self._current is not None:
                logger.info(f"Swapping classifier: {self._current_name} -> {name}")
                self._current.unload()
                self._current = None
                self._current_name = None
                gc.collect()
            
            # Load new classifier
            if name not in self._classifiers:
                raise ValueError(f"Unknown classifier: {name}. Available: {list(self._classifiers.keys())}")
            
            logger.info(f"Loading classifier: {name}")
            classifier = self._classifiers[name]()
            classifier.load()
            
            self._current = classifier
            self._current_name = name
            
            return classifier
    
    def unload_current(self) -> None:
        """Unload the currently loaded classifier."""
        with _registry_lock:
            if self._current is not None:
                logger.info(f"Unloading classifier: {self._current_name}")
                self._current.unload()
                self._current = None
                self._current_name = None
                gc.collect()
    
    @property
    def current_name(self) -> Optional[str]:
        return self._current_name
    
    @property
    def available(self) -> list:
        return list(self._classifiers.keys())


# Global singleton
classifier_registry = ClassifierRegistry()


def get_classifier(name: str) -> BaseClassifier:
    """Convenience function to get a classifier."""
    return classifier_registry.get(name)
