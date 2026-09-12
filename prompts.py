"""
Prompt encoding for the SDXL Image Generation Server.
Handles long prompts (>77 tokens) by chunking and concatenating embeddings.
"""

from typing import Tuple, List, Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline

# Type alias for cached tokenization results
TokenCache = Dict[str, Any]


def encode_long_prompt(
    pipe: 'StableDiffusionXLPipeline',
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    cached_tokens: Optional[Dict[str, TokenCache]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode prompts that are longer than 77 tokens by chunking them.
    
    SDXL has a 77 token limit per text encoder. This function handles longer
    prompts by:
    1. Tokenizing without truncation
    2. Splitting into 75-token chunks (leaving room for BOS/EOS)
    3. Encoding each chunk
    4. Concatenating the embeddings
    5. Generating attention masks to indicate real vs padding tokens
    
    Args:
        pipe: The SDXL pipeline
        prompt: Positive prompt text
        negative_prompt: Negative prompt text
        device: Target device for tensors
        cached_tokens: Optional dict with cached tokenization results from 
                       tokenize_and_check(). Keys: 'prompt' and/or 'negative_prompt'
    
    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds, 
                  pooled_prompt_embeds, negative_pooled_prompt_embeds,
                  prompt_attention_mask, negative_attention_mask)
        
        Attention masks have shape [1, sequence_length] with:
        - 1 for real token positions
        - 0 for padding positions
    """
    cached_tokens = cached_tokens or {}
    
    def process_single_prompt(text: str, cache_key: str, tokenizers, text_encoders):
        """Process a single prompt through both text encoders with caching support."""
        pooled_embeds = None
        encoded_embeds_list = []
        attention_mask_list = []
        
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            # Check cache or tokenize
            if cache_key in cached_tokens and 'input_ids' in cached_tokens[cache_key]:
                input_ids = cached_tokens[cache_key]['input_ids'].to(device)
            else:
                input_ids = tokenizer(
                    [text],
                    padding="longest",
                    return_tensors="pt"
                ).input_ids.to(device)
            
            # If short enough, encode normally
            if input_ids.shape[-1] <= 77:
                outputs = text_encoder(input_ids, output_hidden_states=True)
                encoded_embeds_list.append(outputs.hidden_states[-2])
                
                seq_len = input_ids.shape[-1]
                mask = torch.ones((1, seq_len), device=device, dtype=torch.long)
                attention_mask_list.append(mask)
                
                if text_encoder is pipe.text_encoder_2:
                    pooled_embeds = outputs[0]
                continue
            
            # === LONG PROMPT HANDLING (Vectorized) ===
            chunk_size = 75
            raw_ids = input_ids[0][1:-1]
            total_tokens = raw_ids.shape[0]
            num_chunks = (total_tokens + chunk_size - 1) // chunk_size
            
            chunk_input_ids = torch.full(
                (num_chunks, 77), tokenizer.pad_token_id, dtype=raw_ids.dtype, device=device
            )
            chunk_input_ids[:, 0] = tokenizer.bos_token_id
            
            padded_len = num_chunks * chunk_size
            if total_tokens < padded_len:
                raw_ids_padded = torch.cat([
                    raw_ids,
                    torch.full((padded_len - total_tokens,), tokenizer.pad_token_id, dtype=raw_ids.dtype, device=device)
                ])
            else:
                raw_ids_padded = raw_ids
            
            chunk_input_ids[:, 1:1 + chunk_size] = raw_ids_padded.view(num_chunks, chunk_size)
            
            last_chunk_tokens = total_tokens - (num_chunks - 1) * chunk_size
            if num_chunks > 1:
                chunk_input_ids[:-1, 76] = tokenizer.eos_token_id
            chunk_input_ids[-1, 1 + last_chunk_tokens] = tokenizer.eos_token_id
            
            attention_masks = torch.ones((num_chunks, 77), device=device, dtype=torch.long)
            last_real_count = last_chunk_tokens + 2
            if last_real_count < 77:
                attention_masks[-1, last_real_count:] = 0
            
            combined_mask = attention_masks.view(1, -1)
            
            outputs = text_encoder(chunk_input_ids, output_hidden_states=True)
            chunk_embeddings = outputs.hidden_states[-2]
            
            encoded_embeds_list.append(
                chunk_embeddings.view(1, -1, chunk_embeddings.shape[-1])
            )
            attention_mask_list.append(combined_mask)
            
            if text_encoder is pipe.text_encoder_2:
                pooled_embeds = outputs[0][0].unsqueeze(0)
        
        # Match sequence lengths between encoders
        len_1 = encoded_embeds_list[0].shape[1]
        len_2 = encoded_embeds_list[1].shape[1]
        
        if len_1 != len_2:
            max_len = max(len_1, len_2)
            for i, embed in enumerate(encoded_embeds_list):
                if embed.shape[1] < max_len:
                    pad_size = max_len - embed.shape[1]
                    pad = torch.zeros(
                        (1, pad_size, embed.shape[-1]),
                        device=device,
                        dtype=embed.dtype
                    )
                    encoded_embeds_list[i] = torch.cat([embed, pad], dim=1)
                    mask_pad = torch.zeros((1, pad_size), device=device, dtype=torch.long)
                    attention_mask_list[i] = torch.cat([attention_mask_list[i], mask_pad], dim=1)
        
        final_attention_mask = torch.min(attention_mask_list[0], attention_mask_list[1])
        concat_embeds = torch.cat(encoded_embeds_list, dim=-1)
        
        return concat_embeds, pooled_embeds, final_attention_mask
    
    # Process both prompts
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = process_single_prompt(
        prompt, 'prompt', tokenizers, text_encoders
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds, negative_attention_mask = process_single_prompt(
        negative_prompt, 'negative_prompt', tokenizers, text_encoders
    )
    
    # Match positive/negative sequence lengths for CFG
    pos_len = prompt_embeds.shape[1]
    neg_len = negative_prompt_embeds.shape[1]
    
    if pos_len > neg_len:
        pad = torch.zeros(
            (1, pos_len - neg_len, negative_prompt_embeds.shape[-1]),
            device=device,
            dtype=negative_prompt_embeds.dtype
        )
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)
        # Extend negative attention mask with 0s
        mask_pad = torch.zeros((1, pos_len - neg_len), device=device, dtype=torch.long)
        negative_attention_mask = torch.cat([negative_attention_mask, mask_pad], dim=1)
    elif neg_len > pos_len:
        pad = torch.zeros(
            (1, neg_len - pos_len, prompt_embeds.shape[-1]),
            device=device,
            dtype=prompt_embeds.dtype
        )
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
        # Extend positive attention mask with 0s
        mask_pad = torch.zeros((1, neg_len - pos_len), device=device, dtype=torch.long)
        prompt_attention_mask = torch.cat([prompt_attention_mask, mask_pad], dim=1)
    
    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        prompt_attention_mask,
        negative_attention_mask
    )


def count_tokens(
    pipe: 'StableDiffusionXLPipeline',
    text: str,
    cached_tokens: Optional[TokenCache] = None
) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        pipe: The SDXL pipeline (for tokenizer access)
        text: Text to tokenize
        cached_tokens: Optional cached tokenization result to reuse
    
    Returns:
        Number of tokens
    """
    if cached_tokens and 'input_ids' in cached_tokens:
        return cached_tokens['input_ids'].shape[-1] - 2  # Subtract BOS/EOS
    tokens = pipe.tokenizer(text, return_tensors="pt")
    return tokens.input_ids.shape[-1] - 2  # Subtract BOS/EOS


def is_long_prompt(
    pipe: 'StableDiffusionXLPipeline',
    text: str,
    threshold: int = 75,
    cached_tokens: Optional[TokenCache] = None
) -> bool:
    """
    Check if a prompt exceeds the token threshold.
    
    Args:
        pipe: The SDXL pipeline
        text: Prompt text
        threshold: Token threshold (default 75 to leave room for BOS/EOS)
        cached_tokens: Optional cached tokenization result to reuse
    
    Returns:
        True if prompt exceeds threshold
    """
    return count_tokens(pipe, text, cached_tokens) > threshold


def tokenize_and_check(
    pipe: 'StableDiffusionXLPipeline',
    text: str,
    threshold: int = 75
) -> Tuple[bool, TokenCache]:
    """
    Tokenize a prompt and check if it's long in one operation.
    
    This combines tokenization with the length check so the tokenization
    result can be reused by encode_long_prompt(), avoiding duplicate work.
    
    Args:
        pipe: The SDXL pipeline
        text: Prompt text
        threshold: Token threshold (default 75)
    
    Returns:
        Tuple of (is_long, cached_tokens) where:
        - is_long: True if prompt exceeds threshold
        - cached_tokens: Dict with 'input_ids' and 'token_count' for reuse
    """
    tokens = pipe.tokenizer(text, return_tensors="pt")
    token_count = tokens.input_ids.shape[-1] - 2  # Subtract BOS/EOS
    
    cached = {
        'input_ids': tokens.input_ids,
        'token_count': token_count
    }
    
    return token_count > threshold, cached


def truncate_prompt(
    pipe: 'StableDiffusionXLPipeline',
    text: str,
    max_tokens: int = 75,
    cached_tokens: Optional[TokenCache] = None
) -> str:
    """
    Truncate a prompt to fit within token limit.
    
    Args:
        pipe: The SDXL pipeline
        text: Prompt text
        max_tokens: Maximum tokens (default 75)
        cached_tokens: Optional cached tokenization result (not used for truncation,
                       but accepted for API consistency)
    
    Returns:
        Truncated prompt text
    """
    tokens = pipe.tokenizer(
        text,
        truncation=True,
        max_length=max_tokens + 2,  # +2 for BOS/EOS
        return_tensors="pt"
    )
    
    # Decode back to text
    truncated = pipe.tokenizer.decode(
        tokens.input_ids[0][1:-1],  # Remove BOS/EOS
        skip_special_tokens=True
    )
    
    return truncated
