"""TIPO (Text-to-Image Prompt Optimizer) Engine - converts natural language to Danbooru tags."""

import re
import torch
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "KBlueLeaf/TIPO-200M-FT"
QUALITY_PREFIX = "masterpiece, best quality, high quality, very aesthetic, "
TAG_PATTERN = re.compile(r'^[\w\s,_\-\(\)]+$')
COMMON_TAGS = {'girl', 'boy', 'hair', 'eyes', 'solo', 'smile', 'looking', 'background', 'dress', 'shirt'}


class TIPOEngine:
    """Singleton engine for TIPO prompt enhancement."""
    
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
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
    
    def load(self) -> None:
        """Load TIPO model to CUDA with fp16."""
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=self._device
        )
        self._model.eval()
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def _looks_like_tags(self, prompt: str) -> bool:
        """Check if prompt already looks like Danbooru tags."""
        if ',' not in prompt:
            return False
        parts = [p.strip().lower() for p in prompt.split(',')]
        if len(parts) < 3:
            return False
        tag_like = sum(1 for p in parts if '_' in p or p.replace(' ', '') in COMMON_TAGS or len(p.split()) <= 3)
        return tag_like / len(parts) > 0.5
    
    def enhance(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, 
                prepend_quality: bool = True) -> str:
        """Convert natural language prompt to Danbooru tags."""
        if not prompt or not prompt.strip():
            return QUALITY_PREFIX.rstrip(', ') if prepend_quality else ""
        
        prompt = prompt.strip()
        
        # Skip conversion if already tag-like
        if self._looks_like_tags(prompt):
            return (QUALITY_PREFIX + prompt) if prepend_quality else prompt
        
        if not self.is_loaded:
            self.load()
        
        input_text = f"Natural Language: {prompt} <|sep|> Danbooru Tags: "
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )
        
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract tags after the separator
        if "Danbooru Tags:" in result:
            tags = result.split("Danbooru Tags:")[-1].strip()
        else:
            tags = result.replace(input_text, "").strip()
        
        # Clean up
        tags = tags.split("<|")[0].strip().rstrip(',').strip()
        
        if prepend_quality and tags:
            return QUALITY_PREFIX + tags
        return tags if tags else (QUALITY_PREFIX.rstrip(', ') if prepend_quality else "")
    
    def estimate_vram(self) -> int:
        """Estimate VRAM usage in bytes (~500MB for TIPO-200M)."""
        return 500 * 1024 * 1024


_engine_instance = None

def get_tipo_engine() -> TIPOEngine:
    """Get singleton TIPO engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TIPOEngine()
    return _engine_instance
