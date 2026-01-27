# Operator Analysis Report: complex_rope_attention

## Source File
- **Path**: `sample_operators/complex_rope_attention_structured/`
- **CPU Function**: `rope_attention_cpu` (complex_rope_attention_cpu.py:28)
- **NPU Function**: `rope_attention_npu` (complex_rope_attention_npu.py:31)

## Module Structure
```
complex_rope_attention_structured/
├── __init__.py                      # Package exports
├── complex_rope_attention_utils.py  # Helper functions + constants
├── complex_rope_attention_cpu.py    # CPU reference implementation
└── complex_rope_attention_npu.py    # NPU optimized implementation
```

## Function Signatures

### CPU Reference Implementation
```python
def rope_attention_cpu(
    query: torch.Tensor,                           # (batch, num_heads, seq_len_q, head_dim)
    key: torch.Tensor,                             # (batch, num_heads, seq_len_k, head_dim)
    value: torch.Tensor,                           # (batch, num_heads, seq_len_k, head_dim)
    cos_cached: torch.Tensor,                      # (max_seq_len, head_dim // 2)
    sin_cached: torch.Tensor,                      # (max_seq_len, head_dim // 2)
    attention_mask: Optional[torch.Tensor] = None, # (seq_len_q, seq_len_k) or (batch, heads, seq_q, seq_k)
    key_padding_mask: Optional[torch.Tensor] = None, # (batch, seq_len_k)
    is_causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    training: bool = False,
    use_flash: bool = False,                       # Ignored on CPU
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns: (output, attention_weights)"""
```

### NPU Implementation
```python
def rope_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    training: bool = False,
    use_flash: bool = True,                        # Default True on NPU
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns: (output, attention_weights)"""
```

## Parameter Analysis

| # | Parameter | Type | Default | Category | Test Priority | Notes |
|---|-----------|------|---------|----------|---------------|-------|
| 1 | `query` | Tensor | required | **Input Tensor** | **HIGH** | Shape: (batch, num_heads, seq_len_q, head_dim) |
| 2 | `key` | Tensor | required | **Input Tensor** | **HIGH** | Shape: (batch, num_heads, seq_len_k, head_dim) |
| 3 | `value` | Tensor | required | **Input Tensor** | **HIGH** | Shape: (batch, num_heads, seq_len_k, head_dim) |
| 4 | `cos_cached` | Tensor | required | **Input Tensor** | **HIGH** | Precomputed RoPE cos values |
| 5 | `sin_cached` | Tensor | required | **Input Tensor** | **HIGH** | Precomputed RoPE sin values |
| 6 | `attention_mask` | Tensor | None | **Shape/Input** | **HIGH** | Optional: 2D or 4D mask |
| 7 | `key_padding_mask` | Tensor | None | **Shape/Input** | **MEDIUM** | Optional: (batch, seq_len_k) boolean mask |
| 8 | `is_causal` | bool | False | **Control Flag** | **HIGH** | Enables causal masking (decoder-style) |
| 9 | `dropout_p` | float | 0.0 | **Algorithm** | **MEDIUM** | Only applied when training=True |
| 10 | `softmax_scale` | float | None | **Algorithm** | **MEDIUM** | Defaults to 1/sqrt(head_dim) |
| 11 | `training` | bool | False | **Control Flag** | **LOW** | Enables dropout |
| 12 | `use_flash` | bool | False/True | **Control Flag** | **HIGH** | CPU=False, NPU=True - selects algorithm path |

## Inferred Operator Type

- **Type**: `attention`
- **Reasoning**: This is a **Multi-Head Attention with Rotary Position Embedding (RoPE)** operator. It computes scaled dot-product attention between query, key, and value tensors, with the addition of rotary position embeddings applied to Q and K before the attention computation.

## Key Implementation Details

### Algorithm Variants
| Path | Condition | Description |
|------|-----------|-------------|
| **Naive Attention** | Default | Standard QK^T @ V computation |
| **Flash Attention** | NPU + seq_len ≥ 1024 + supported dtype | Optimized kernel (placeholder) |

### Supported Configurations
| Property | Values |
|----------|--------|
| **Supported dtypes** | float32, float16, bfloat16 |
| **Supported head dims** | 64, 128, 256 |
| **Flash threshold** | seq_len ≥ 1024 |
| **Max seq length** | 8192 |

### Helper Functions
| Function | Purpose |
|----------|---------|
| `_validate_inputs()` | Shape and dtype validation |
| `_compute_rope_frequencies()` | Precompute RoPE frequency basis |
| `_apply_rotary_embedding()` | Apply rotation to Q/K tensors |
| `_create_causal_mask()` | Generate causal (lower-triangular) mask |
| `_merge_attention_masks()` | Combine user and causal masks |

## Shape Analysis

### Input Shapes
| Tensor | Shape | Description |
|--------|-------|-------------|
| `query` | `(B, H, Lq, D)` | Batch, Heads, Query Length, Head Dim |
| `key` | `(B, H, Lk, D)` | Batch, Heads, Key Length, Head Dim |
| `value` | `(B, H, Lk, D)` | Batch, Heads, Key Length, Head Dim |
| `cos_cached` | `(max_seq_len, D/2)` | Precomputed cosine values |
| `sin_cached` | `(max_seq_len, D/2)` | Precomputed sine values |

### Output Shapes
| Tensor | Shape | Description |
|--------|-------|-------------|
| `output` | `(B, H, Lq, D)` | Attention output |
| `attention_weights` | `(B, H, Lq, Lk)` | Attention scores (after softmax) |

## Observations

### Data Types
- Supported: `torch.float32`, `torch.float16`, `torch.bfloat16`
- Flash attention requires `float16` or `bfloat16`

### Test Dimensions (from original test code)
```python
batch_size = 2
num_heads = 8
seq_len = 128
head_dim = 64
```

### Special Considerations
1. **Cross-attention support**: `seq_len_q` and `seq_len_k` can differ
2. **Causal masking**: Commonly used in autoregressive models (GPT-style)
3. **RoPE is position-dependent**: Unlike absolute position embeddings, RoPE mixes positions into the query/key values
4. **Flash attention placeholder**: The NPU flash attention currently falls back to naive implementation
5. **Device handling**: NPU implementation moves cached cos/sin to NPU if needed

### Differences Between CPU and NPU
| Aspect | CPU | NPU |
|--------|-----|-----|
| `use_flash` default | False | True |
| Flash behavior | Ignored | Conditional on seq_len, dtype, head_dim |
| Device transfer | None | Moves cos/sin to NPU |

## Questions for User

1. **Flash attention**: The NPU flash attention is a placeholder (falls back to naive). Should we test the actual flash attention kernel when available, or only the naive path?

2. **Dtype priorities**: Should we prioritize testing float16/bfloat16 (for flash attention path) or include float32 as well?

3. **Causal vs non-causal**: Should both be tested, or focus on one (causal is more common in decoder models)?

4. **Mask combinations**: Should we test:
   - No mask
   - Causal only
   - User mask only
   - Both masks combined

---

**Analysis completed**: 2026-01-28
**Next step**: Use `/plan-operator-test` to generate test configurations
