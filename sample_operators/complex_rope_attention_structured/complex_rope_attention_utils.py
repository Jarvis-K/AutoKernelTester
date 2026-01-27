"""
Complex RoPE Attention - Utility Functions Module

Contains helper functions and constants used by both CPU and NPU implementations.
"""

import torch
import math
from typing import Optional

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
            # Expand to (batch_size, num_heads, seq_q, seq_k) for broadcasting
            return attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        return attention_mask

    # Merge both masks
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    causal_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
    return attention_mask + causal_expanded
