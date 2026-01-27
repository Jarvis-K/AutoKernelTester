"""
Complex RoPE Attention - Restructured Module

This package contains the restructured complex_rope_attention operator,
split into modular components for easier testing and analysis.

Modules:
    - complex_rope_attention_utils: Helper functions and constants
    - complex_rope_attention_cpu: CPU reference implementation
    - complex_rope_attention_npu: NPU optimized implementation
"""

from .complex_rope_attention_cpu import rope_attention_cpu
from .complex_rope_attention_npu import rope_attention_npu
from .complex_rope_attention_utils import (
    _compute_rope_frequencies,
    _validate_inputs,
    _apply_rotary_embedding,
    _create_causal_mask,
    _merge_attention_masks,
    DEFAULT_ATTENTION_DROPOUT,
    DEFAULT_SOFTMAX_SCALE,
    MAX_SEQ_LEN,
    FLASH_ATTENTION_THRESHOLD,
    ROPE_THETA,
    SUPPORTED_DTYPES,
    SUPPORTED_HEAD_DIMS,
)

__all__ = [
    'rope_attention_cpu',
    'rope_attention_npu',
    '_compute_rope_frequencies',
    '_validate_inputs',
    '_apply_rotary_embedding',
    '_create_causal_mask',
    '_merge_attention_masks',
    'DEFAULT_ATTENTION_DROPOUT',
    'DEFAULT_SOFTMAX_SCALE',
    'MAX_SEQ_LEN',
    'FLASH_ATTENTION_THRESHOLD',
    'ROPE_THETA',
    'SUPPORTED_DTYPES',
    'SUPPORTED_HEAD_DIMS',
]
