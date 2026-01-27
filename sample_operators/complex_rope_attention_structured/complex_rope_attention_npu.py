"""
Complex RoPE Attention - NPU Implementation Module

NPU optimized implementation of RoPE attention with flash attention support.
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
    FLASH_ATTENTION_THRESHOLD,
    SUPPORTED_HEAD_DIMS,
    SUPPORTED_DTYPES,
)


# =============================================================================
# NPU IMPLEMENTATION
# =============================================================================

def rope_attention_npu(
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
    use_flash: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU optimized implementation of RoPE attention.

    Uses flash attention when available and seq_len > FLASH_ATTENTION_THRESHOLD.
    Falls back to naive implementation otherwise.

    Args:
        Same as rope_attention_cpu

    Returns:
        Same as rope_attention_cpu
    """
    try:
        import torch_npu
        HAS_NPU = True
    except ImportError:
        HAS_NPU = False

    # Validate inputs
    _validate_inputs(query, key, value, attention_mask)

    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]

    # Move cached values to NPU if needed
    if HAS_NPU and query.device.type == 'npu':
        if cos_cached.device.type != 'npu':
            cos_cached = cos_cached.to(query.device)
        if sin_cached.device.type != 'npu':
            sin_cached = sin_cached.to(query.device)

    # Apply rotary position embeddings
    query_rope = _apply_rotary_embedding(query, cos_cached, sin_cached)
    key_rope = _apply_rotary_embedding(key, cos_cached, sin_cached)

    # Decide whether to use flash attention
    use_flash_impl = (
        use_flash
        and HAS_NPU
        and seq_len_q >= FLASH_ATTENTION_THRESHOLD
        and head_dim in SUPPORTED_HEAD_DIMS
        and query.dtype in [torch.float16, torch.bfloat16]
    )

    if use_flash_impl:
        # Flash attention path (NPU optimized)
        output, attention_weights = _npu_flash_attention(
            query_rope, key_rope, value,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=dropout_p if training else 0.0,
            softmax_scale=softmax_scale,
        )
    else:
        # Naive attention path
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(query_rope, key_rope.transpose(-2, -1)) * softmax_scale

        # Build combined mask
        causal_mask = None
        if is_causal:
            causal_mask = _create_causal_mask(seq_len_q, query.device, query.dtype)
            if seq_len_k != seq_len_q:
                full_causal = torch.zeros(seq_len_q, seq_len_k, device=query.device, dtype=query.dtype)
                full_causal[:, :seq_len_q] = causal_mask
                causal_mask = full_causal

        combined_mask = _merge_attention_masks(attention_mask, causal_mask, batch_size, num_heads)

        if combined_mask is not None:
            attention_scores = attention_scores + combined_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        if training and dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

        output = torch.matmul(attention_weights, value)

    return output, attention_weights


def _npu_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU Flash Attention implementation.

    This is a placeholder - in real implementation, this would call
    torch_npu.npu_fusion_attention or similar optimized kernel.
    """
    try:
        import torch_npu

        batch_size, num_heads, seq_len_q, head_dim = query.shape
        seq_len_k = key.shape[2]

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        # Convert to format expected by NPU flash attention
        # (batch, heads, seq, dim) -> (batch, seq, heads, dim)
        query_t = query.transpose(1, 2).contiguous()
        key_t = key.transpose(1, 2).contiguous()
        value_t = value.transpose(1, 2).contiguous()

        # Call NPU flash attention (placeholder - actual API may differ)
        # output = torch_npu.npu_fusion_attention(
        #     query_t, key_t, value_t,
        #     head_num=num_heads,
        #     input_layout="BNSD",
        #     atten_mask=attention_mask,
        #     scale=softmax_scale,
        #     keep_prob=1.0 - dropout_p,
        # )

        # Fallback to naive for now
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale

        if is_causal:
            causal_mask = _create_causal_mask(seq_len_q, query.device, query.dtype)
            attention_scores = attention_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attention_weights = F.softmax(attention_scores, dim=-1)

        if dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights

    except ImportError:
        raise RuntimeError("torch_npu is required for NPU flash attention")
