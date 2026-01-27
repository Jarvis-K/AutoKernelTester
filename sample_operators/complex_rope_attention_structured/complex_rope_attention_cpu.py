"""
Complex RoPE Attention - CPU Implementation Module

CPU reference implementation of RoPE attention.
Uses utilities from the utils module.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import utilities from shared module
from .complex_rope_attention_utils import (
    _validate_inputs,
    _apply_rotary_embedding,
    _create_causal_mask,
    _merge_attention_masks,
    DEFAULT_ATTENTION_DROPOUT,
    DEFAULT_SOFTMAX_SCALE,
)


# =============================================================================
# CPU REFERENCE IMPLEMENTATION
# =============================================================================

def rope_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = DEFAULT_ATTENTION_DROPOUT,
    softmax_scale: Optional[float] = DEFAULT_SOFTMAX_SCALE,
    training: bool = False,
    use_flash: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation of RoPE attention.

    Args:
        query: (batch, num_heads, seq_len_q, head_dim)
        key: (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        cos_cached: (max_seq_len, head_dim // 2) - precomputed cos for RoPE
        sin_cached: (max_seq_len, head_dim // 2) - precomputed sin for RoPE
        attention_mask: Optional attention mask
        key_padding_mask: Optional key padding mask
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        softmax_scale: Custom softmax scale (default: 1/sqrt(head_dim))
        training: Whether in training mode
        use_flash: Ignored on CPU (always uses naive implementation)

    Returns:
        output: (batch, num_heads, seq_len_q, head_dim)
        attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
    """
    # Validate inputs
    _validate_inputs(query, key, value, attention_mask)

    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]

    # Apply rotary position embeddings
    query_rope = _apply_rotary_embedding(query, cos_cached, sin_cached)
    key_rope = _apply_rotary_embedding(key, cos_cached, sin_cached)

    # Compute attention scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: (batch, heads, seq_q, seq_k)
    attention_scores = torch.matmul(query_rope, key_rope.transpose(-2, -1)) * softmax_scale

    # Build combined mask
    causal_mask = None
    if is_causal:
        causal_mask = _create_causal_mask(seq_len_q, query.device, query.dtype)
        if seq_len_k != seq_len_q:
            # For cross-attention or different key length
            full_causal = torch.zeros(seq_len_q, seq_len_k, device=query.device, dtype=query.dtype)
            full_causal[:, :seq_len_q] = causal_mask
            causal_mask = full_causal

    combined_mask = _merge_attention_masks(attention_mask, causal_mask, batch_size, num_heads)

    # Apply mask
    if combined_mask is not None:
        attention_scores = attention_scores + combined_mask

    # Apply key padding mask
    if key_padding_mask is not None:
        # key_padding_mask: (batch, seq_k) - True means masked
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))

    # Softmax
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Apply dropout
    if training and dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

    # Compute output
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
