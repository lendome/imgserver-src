"""Embedding manager for textual inversion embeddings."""
import hashlib
import logging
import requests
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Singleton manager for textual inversion embeddings."""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.cache_dir = Path("./embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_tokens: list[str] = []
        self._initialized = True
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache path from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        filename = Path(urlparse(url).path).name or f"embedding_{url_hash}"
        if not any(filename.endswith(ext) for ext in ['.safetensors', '.pt', '.bin']):
            filename = f"{filename}.safetensors"
        return self.cache_dir / f"{url_hash}_{filename}"
    
    def download_embedding(self, url: str) -> Path:
        """Download embedding to cache, return path."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            logger.debug(f"Embedding cached: {cache_path}")
            return cache_path
        
        logger.info(f"Downloading embedding: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
        return cache_path
    
    def load_embeddings(self, pipeline, embeddings: list[dict]) -> None:
        """Load textual inversion embeddings into pipeline.
        
        Args:
            pipeline: Diffusers pipeline with text encoder(s)
            embeddings: List of {"url": str, "token": str, "strength": float}
        """
        for emb in embeddings:
            url = emb["url"]
            token = emb["token"]
            
            try:
                path = self.download_embedding(url)
                
                # Handle SDXL dual encoders
                if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                    try:
                        pipeline.load_textual_inversion(path, token=token)
                    except Exception:
                        # Try loading into each encoder separately for some formats
                        pipeline.load_textual_inversion(
                            path, token=token, text_encoder=pipeline.text_encoder,
                            tokenizer=pipeline.tokenizer
                        )
                else:
                    pipeline.load_textual_inversion(path, token=token)
                
                self._loaded_tokens.append(token)
                logger.info(f"Loaded embedding: {token}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding {token}: {e}")
    
    def clear_embeddings(self, pipeline) -> None:
        """Clear loaded embeddings (limited support in diffusers)."""
        if self._loaded_tokens:
            logger.warning(
                f"Diffusers lacks clean embedding unload. "
                f"Tokens still registered: {self._loaded_tokens}. "
                f"Consider recreating pipeline for full cleanup."
            )
        self._loaded_tokens.clear()
    
    def get_loaded_tokens(self) -> list[str]:
        """Return list of currently loaded tokens."""
        return self._loaded_tokens.copy()
