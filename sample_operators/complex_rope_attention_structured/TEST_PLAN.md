# Test Plan: complex_rope_attention

## Test Summary

| Category | Count | Details |
|----------|-------|---------|
| **Batch Sizes** | 7 | `[1, 2, 4, 8, 16, 32, 48]` (comprehensive coverage) |
| **Sequence Lengths** | 5 | `[64, 128, 512, 1024, 2048]` (below + above flash threshold) |
| **Head Dimensions** | 3 | `[64, 128, 256]` (all supported) |
| **Num Heads** | 4 | `[8, 16, 32, 40]` (small to large models) |
| **Data Types** | 3 | `[float32, float16, bfloat16]` |
| **Value Patterns** | 6 | `[random, zeros, ones, very_small, very_large, mixed_sign]` |
| **Causal Modes** | 2 | `[causal=True, causal=False]` |
| **Mask Combinations** | 4 | `[none, causal_only, user_only, combined]` |
| **Flash Paths** | 2 | `[naive, flash]` |
| **Total Tests** | ~**2,016** | Before cross-product reduction |

---

## Test Configurations

### 1. Input Tensor Shapes

Based on LLM model size guidelines:

| Model Size | batch | num_heads | seq_len | head_dim | Description |
|------------|-------|-----------|---------|----------|-------------|
| **Small (125M)** | 1-8 | 8-12 | 64, 128 | 64 | Lightweight models |
| **Medium (1-3B)** | 1-4 | 16-20 | 128, 512 | 128 | Standard models |
| **Large (7-13B)** | 1-2 | 32-40 | 1024, 2048 | 128 | Large models |

#### Full Shape Matrix

```python
# Batch sizes (comprehensive)
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48]

# Sequence lengths (below + above flash threshold)
SEQ_LENS = [64, 128, 512, 1024, 2048]

# Head dimensions (all supported)
HEAD_DIMS = [64, 128, 256]

# Number of attention heads
NUM_HEADS = [8, 16, 32, 40]

# Cross-attention: seq_len_q ≠ seq_len_k
CROSS_ATTENTION_SHAPES = [
    (128, 256),   # query=128, key=256
    (512, 1024),  # query=512, key=1024
    (256, 128),   # query=256, key=128
]
```

#### Representative Test Shapes

| Test Case | Shape (B, H, Lq, Lk, D) | Purpose |
|-----------|------------------------|---------|
| Minimal | `(1, 8, 64, 64, 64)` | Small scale, fast iteration |
| Standard | `(2, 16, 128, 128, 128)` | Medium model baseline |
| Large | `(1, 32, 1024, 1024, 128)` | Above flash threshold |
| XLarge | `(1, 40, 2048, 2048, 128)` | Long context |
| Cross | `(2, 16, 128, 256, 128)` | Cross-attention |
| Batched | `(48, 8, 128, 128, 64)` | Max batch size |

---

### 2. Data Types

| Dtype | Priority | Test Count | Notes |
|-------|----------|------------|-------|
| **float16** | HIGH | ~672 | Required for flash attention path |
| **bfloat16** | HIGH | ~672 | Required for flash attention path |
| **float32** | MEDIUM | ~672 | Baseline reference, no flash |

**Dtype Test Distribution:**
- Flash attention tests: Only float16/bfloat16
- Naive attention tests: All dtypes

---

### 3. Value Patterns

```python
PATTERNS = [
    ("random", "Standard normal distribution"),
    ("zeros", "All zeros - edge case"),
    ("ones", "All ones - uniform values"),
    ("very_small", "Values ~1e-6"),
    ("very_large", "Values ~1e6"),
    ("mixed_sign", "Positive and negative values"),
]
```

---

### 4. Algorithm Parameters

| Parameter | Test Values | Priority |
|-----------|-------------|----------|
| **is_causal** | `[True, False]` | HIGH |
| **use_flash** | `[True, False]` | HIGH |
| **dropout_p** | `[0.0, 0.1]` (only when training=True) | MEDIUM |
| **softmax_scale** | `[None, 0.5, 1.0/sqrt(head_dim)]` | MEDIUM |
| **training** | `[False, True]` | LOW |

---

### 5. Mask Combinations (All Requested)

| Mask Type | attention_mask | key_padding_mask | is_causal | Test Count |
|-----------|----------------|------------------|-----------|------------|
| **No mask** | None | None | False | ~504 |
| **Causal only** | None | None | True | ~504 |
| **User mask only** | 2D tensor | None | False | ~504 |
| **Combined** | 2D tensor | None | True | ~504 |

**Note:** key_padding_mask tests add additional ~252 cases

---

## Tolerance Settings

Based on operator type `attention`:

| Dtype | rtol | atol | Notes |
|-------|------|------|-------|
| **float32** | 1e-3 | 1e-4 | Standard precision |
| **float16** | 1e-2 | 1e-3 | Reduced precision for FP16 |
| **bfloat16** | 5e-2 | 1e-2 | Most lenient for BF16 |

**Attention-specific considerations:**
- Softmax can amplify small differences
- RoPE rotation introduces trigonometric computations
- Gradient differences accumulate in deep attention stacks

---

## Test Case Breakdown

### Core Test Matrix

```
                × 7 batch sizes
                × 5 sequence lengths
                × 3 head dimensions
                × 4 num_heads
                × 3 dtypes
                × 2 causal modes
                × 4 mask combinations
                × 2 flash paths
                × 6 value patterns
                = 241,920 (theoretical max)
```

### Optimized Test Selection

To keep testing practical while maintaining coverage:

| Test Suite | Cases | Focus |
|------------|-------|-------|
| **Quick Smoke** | ~50 | Minimal shapes, float32 only, no masks |
| **Dtype Coverage** | ~100 | All dtypes, one shape per dtype |
| **Shape Sweep** | ~200 | All shape combinations, float16 only |
| **Mask Matrix** | ~100 | All 4 mask combinations, 2 shapes |
| **Flash vs Naive** | ~50 | Below/above 1024 threshold, float16 |
| **Edge Cases** | ~50 | Zeros, ones, very small/large |
| **Comprehensive** | ~550 | Full coverage for validation |
| **Production** | ~2,000 | Full matrix for final validation |

**Recommended default:** **Comprehensive (~550 tests)**

---

## Test Execution Strategy

### Phase 1: Quick Smoke (~50 tests, ~30 seconds)
- Batch sizes: `[1, 2]`
- Sequence lengths: `[64, 128]`
- Head dims: `[64]`
- Dtypes: `[float32]`
- No masks, causal=False
- Pattern: random only

**Purpose:** Verify basic functionality before extensive testing

### Phase 2: Flash Threshold Detection (~100 tests, ~2 minutes)
- Focus: seq_len in `[512, 1024, 2048]`
- Dtypes: `[float16, bfloat16]`
- Compare: use_flash=True vs False
- Verify: same output (within tolerance)

**Purpose:** Ensure flash/naive path equivalence

### Phase 3: Comprehensive (~550 tests, ~10 minutes)
- All batch sizes: `[1, 2, 4, 8, 16, 32, 48]`
- All sequence lengths: `[64, 128, 512, 1024, 2048]`
- All head dims: `[64, 128, 256]`
- All dtypes: `[float32, float16, bfloat16]`
- All mask combinations
- All patterns

**Purpose:** Full validation before production use

---

## Edge Cases

| Case | Description | Expected Behavior |
|------|-------------|-------------------|
| Single element | `(1, 1, 1, 1)` | Should work correctly |
| Non-contiguous | Strided slices | Should handle or error gracefully |
| Very large values | `~1e6` | No overflow/NaN |
| Very small values | `~1e-6` | No underflow/zero grad |
| seq_len > max_seq_len | `> 8192` | Error or graceful handling |
| Unsupported head_dim | `32, 512` | Should fall back or error |
| Cross-attention | `Lq ≠ Lk` | Correct masking |

---

## Special Considerations

### RoPE-Specific

1. **Frequency caching:** cos_cached and sin_cached must be precomputed with correct max_seq_len
2. **Position slicing:** cos[:seq_len] and sin[:seq_len] must work correctly
3. **Trigonometric precision:** Rotations may differ slightly across devices

### Flash Attention

1. **Threshold:** seq_len ≥ 1024 triggers flash path (when dtype supported)
2. **Dtype restriction:** Only float16/bfloat16 eligible
3. **Head dimension:** Must be in `[64, 128, 256]`
4. **Placeholder fallback:** Current implementation uses naive path

### Cross-Attention

1. **Sequence length mismatch:** Lq and Lk can be different
2. **Causal mask shape:** Must handle rectangular case (Lq × Lk)

---

## Estimated Runtime

| Test Suite | Tests | Est. Time | Notes |
|------------|-------|-----------|-------|
| Quick Smoke | 50 | ~30 sec | Fast iteration during dev |
| Flash Detection | 100 | ~2 min | Verify flash/naive equivalence |
| Comprehensive | 550 | ~10 min | Recommended for PR validation |
| Production | 2,000 | ~35 min | Full matrix before release |

**Assumptions:**
- CPU-only testing (no NPU hardware)
- Average test case: ~1 second
- Includes verification overhead

---

## Test Script Structure

```python
test_configurations = [
    # Quick smoke tests
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64,
     "dtype": torch.float32, "causal": False, "masks": "none"},

    # Flash threshold tests
    {"batch": 1, "heads": 16, "seq_q": 1024, "seq_k": 1024, "head_dim": 128,
     "dtype": torch.float16, "causal": False, "use_flash": True},
    {"batch": 1, "heads": 16, "seq_q": 1024, "seq_k": 1024, "head_dim": 128,
     "dtype": torch.float16, "causal": False, "use_flash": False},

    # Mask combinations
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 128,
     "dtype": torch.float16, "causal": True, "masks": "causal_only"},
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 128,
     "dtype": torch.float16, "causal": False, "masks": "user_only"},

    # Cross-attention
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 256, "head_dim": 128,
     "dtype": torch.float16, "causal": False, "masks": "none"},

    # Edge cases
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64,
     "dtype": torch.float32, "pattern": "zeros"},
]
```

---

## Success Criteria

A test configuration is considered **PASS** if:

1. **Output shape matches** expected `(B, H, Lq, D)`
2. **Values within tolerance** (per dtype settings above)
3. **No NaN/Inf** in outputs
4. **Attention weights valid** (sum to ~1.0 per query position)
5. **Masks applied correctly** (masked positions have ~0 attention)

**Overall pass rate target:** ≥ 95%

---

## Next Steps

After approval of this test plan:

1. Use `/execute-operator-test` to generate and run the test script
2. Review CSV results and pass/fail breakdown
3. Investigate any failing configurations
4. Adjust tolerances if needed (with justification)

---

**Test Plan created:** 2026-01-28
**Status:** ✅ Ready for approval
**Recommended:** Comprehensive suite (~550 tests, ~10 minutes)
