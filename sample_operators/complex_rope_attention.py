"""
Complex Multi-Head Attention Operator with Rotary Position Embedding (RoPE)

This is a complex operator that should trigger the restructure skill:
- 400+ lines of code
- 7 input tensors
- 10+ parameters
- 5 helper functions
- Multiple algorithm variants (flash attention vs naive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

DEFAULT_ATTENTION_DROPOUT = 0.0
DEFAULT_SOFTMAX_SCALE = None
MAX_SEQ_LEN = 8192
FLASH_ATTENTION_THRESHOLD = 1024  # Use flash attention for seq_len > this
ROPE_THETA = 10000.0
EPS = 1e-6

SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SUPPORTED_HEAD_DIMS = [64, 128, 256]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """Validate input tensor shapes and dtypes."""
    if query.dim() != 4:
        raise ValueError(f"Query must be 4D, got {query.dim()}D")
    if key.dim() != 4:
        raise ValueError(f"Key must be 4D, got {key.dim()}D")
    if value.dim() != 4:
        raise ValueError(f"Value must be 4D, got {value.dim()}D")
    
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if key.shape[0] != batch_size or value.shape[0] != batch_size:
        raise ValueError("Batch sizes must match")
    if key.shape[1] != num_heads or value.shape[1] != num_heads:
        raise ValueError("Number of heads must match")
    if key.shape[3] != head_dim or value.shape[3] != head_dim:
        raise ValueError("Head dimensions must match")
    
    if attention_mask is not None:
        if attention_mask.dim() not in [2, 4]:
            raise ValueError(f"Attention mask must be 2D or 4D, got {attention_mask.dim()}D")


def _compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = ROPE_THETA,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute rotary position embedding frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)
    return freqs


def _apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    # x: (batch, heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim // 2)
    batch, heads, seq_len, head_dim = x.shape
    
    x_reshape = x.view(batch, heads, seq_len, head_dim // 2, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
    
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    return torch.stack([out1, out2], dim=-1).view(batch, heads, seq_len, head_dim)


def _create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float('-inf'),
        diagonal=1
    )
    return mask


def _merge_attention_masks(
    attention_mask: Optional[torch.Tensor],
    causal_mask: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
) -> Optional[torch.Tensor]:
    """Merge user-provided mask with causal mask."""
    if attention_mask is None and causal_mask is None:
        return None
    
    if attention_mask is None:
        return causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    
    if causal_mask is None:
        if attention_mask.dim() == 2:
            return attention_mask.unsqueeze(1).unsqueeze(1)
        return attention_mask
    
    # Merge both masks
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    
    causal_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
    return attention_mask + causal_expanded


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


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    # Simple test
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    
    # Create inputs
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Compute RoPE frequencies
    freqs = _compute_rope_frequencies(head_dim, MAX_SEQ_LEN)
    cos_cached = torch.cos(freqs)
    sin_cached = torch.sin(freqs)
    
    # Run CPU reference
    output_cpu, weights_cpu = rope_attention_cpu(
        query, key, value, cos_cached, sin_cached, is_causal=True
    )
    
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output_cpu.shape}")
    print(f"Weights shape: {weights_cpu.shape}")
    print("CPU reference test passed!")
