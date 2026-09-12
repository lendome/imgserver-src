"""
Prompt Preprocessor for filtering banned words from user prompts.
Provides safety mechanisms to remove potentially harmful content.
"""

import re
import logging
from typing import List, Tuple
from threading import Lock

logger = logging.getLogger(__name__)


class PromptPreprocessor:
    """
    Filters banned words and phrases from prompts.
    
    Uses regex with word boundaries to ensure accurate matching.
    Thread-safe implementation for concurrent access.
    """
    
    # Default banned words list
    DEFAULT_BANNED_WORDS = [
        "younger",
        "young",
        "underage",
        "minor",
        "child",
        "kid",
        "loli",
        "shota",
        "preteen",
        "teen",
        "teenage",
        "adolescent",
        "juvenile",
    ]
    
    def __init__(self, banned_words: List[str] = None):
        """
        Initialize the PromptPreprocessor.
        
        Args:
            banned_words: List of words to ban. If None, uses DEFAULT_BANNED_WORDS.
        """
        self._lock = Lock()
        
        if banned_words is None:
            self.banned_words = self.DEFAULT_BANNED_WORDS.copy()
        else:
            self.banned_words = [word.lower().strip() for word in banned_words]
        
        logger.info(f"PromptPreprocessor initialized with {len(self.banned_words)} banned words")
    
    def _build_pattern(self) -> re.Pattern:
        """
        Build a regex pattern for all banned words.
        
        Uses word boundaries (\b) to match whole words only.
        
        Returns:
            Compiled regex pattern for banned words.
        """
        if not self.banned_words:
            return None
        
        # Escape special regex characters and join with OR
        escaped_words = [re.escape(word) for word in self.banned_words]
        pattern_str = r"\b(" + "|".join(escaped_words) + r")\b"
        return re.compile(pattern_str, re.IGNORECASE)
    
    def filter_prompt(self, prompt: str) -> str:
        """
        Remove banned words from the prompt.
        
        Replaces banned words with empty string and cleans up extra spaces.
        
        Args:
            prompt: The prompt text to filter.
            
        Returns:
            Filtered prompt with banned words removed.
        """
        if not prompt:
            return prompt
        
        with self._lock:
            pattern = self._build_pattern()
            if pattern is None:
                return prompt
            
            # Replace banned words with empty string
            filtered = pattern.sub("", prompt)
            
            # Clean up extra spaces
            filtered = re.sub(r"\s+", " ", filtered).strip()
            
            logger.debug(f"Filtered prompt. Original length: {len(prompt)}, Filtered length: {len(filtered)}")
            
            return filtered
    
    def contains_banned(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Check if prompt contains banned words.
        
        Args:
            prompt: The prompt text to check.
            
        Returns:
            Tuple of (has_banned_words, list_of_banned_words_found)
        """
        if not prompt:
            return False, []
        
        with self._lock:
            pattern = self._build_pattern()
            if pattern is None:
                return False, []
            
            matches = pattern.findall(prompt)
            found_words = list(set(word.lower() for word in matches))
            
            if found_words:
                logger.warning(f"Prompt contains banned words: {found_words}")
            
            return len(found_words) > 0, found_words
    
    def add_banned_word(self, word: str) -> None:
        """
        Add a word to the banned words list.
        
        Args:
            word: Word to add to ban list.
        """
        with self._lock:
            word_lower = word.lower().strip()
            if word_lower and word_lower not in self.banned_words:
                self.banned_words.append(word_lower)
                logger.info(f"Added banned word: '{word_lower}'")
            elif word_lower in self.banned_words:
                logger.debug(f"Word '{word_lower}' already in banned list")
    
    def remove_banned_word(self, word: str) -> None:
        """
        Remove a word from the banned words list.
        
        Args:
            word: Word to remove from ban list.
        """
        with self._lock:
            word_lower = word.lower().strip()
            if word_lower in self.banned_words:
                self.banned_words.remove(word_lower)
                logger.info(f"Removed banned word: '{word_lower}'")
            else:
                logger.debug(f"Word '{word_lower}' not found in banned list")
    
    def get_banned_words(self) -> List[str]:
        """
        Get the current list of banned words.
        
        Returns:
            Copy of the banned words list.
        """
        with self._lock:
            return self.banned_words.copy()
    
    def set_banned_words(self, banned_words: List[str]) -> None:
        """
        Replace the entire banned words list.
        
        Args:
            banned_words: New list of banned words.
        """
        with self._lock:
            self.banned_words = [word.lower().strip() for word in banned_words]
            logger.info(f"Banned words list updated with {len(self.banned_words)} words")
