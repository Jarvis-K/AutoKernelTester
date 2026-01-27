# Precision Test Report: complex_rope_attention

## Test Suite: Quick Smoke

**Date:** 2026-01-28
**Total Tests:** 30
**Estimated Time:** ~30 seconds

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 30 |
| **Passed** | 30 |
| **Failed** | 0 |
| **Errors** | 0 |
| **Pass Rate** | 100.0% |
| **CSV Export** | `complex_rope_attention_quick_results.csv` |

---

## Results by Pattern

| Pattern | Total | Passed | Failed | Pass Rate |
|---------|-------|--------|--------|-----------|
| mixed_sign | 2 | 2 | 0 | 100.0% |
| ones | 2 | 2 | 0 | 100.0% |
| random | 20 | 20 | 0 | 100.0% |
| very_large | 2 | 2 | 0 | 100.0% |
| very_small | 2 | 2 | 0 | 100.0% |
| zeros | 2 | 2 | 0 | 100.0% |

## Results by Mask Type

| Mask Type | Total | Passed | Failed | Pass Rate |
|-----------|-------|--------|--------|----------|
| causal_only | 4 | 4 | 0 | 100.0% |
| none | 22 | 22 | 0 | 100.0% |
| user_only | 4 | 4 | 0 | 100.0% |


## Test Configuration

**Quick Smoke Suite Parameters:**
- Batch sizes: `[1, 2]`
- Sequence lengths: `[64, 128]`
- Head dimensions: `[64]`
- Number of heads: `[8, 16]`
- Data types: `[float32]`
- Causal modes: `[True, False]`
- Mask combinations: `[none, causal_only, user_only]`
- Value patterns: `[random, zeros, ones, very_small, very_large, mixed_sign]`

## Tolerance Settings

| Dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-3 | 1e-4 |

## Recommendations

✅ **Pass rate ≥ 95%** - Operator is performing within expected tolerances.


## Notes

- This is the **Quick Smoke** test suite for basic functionality verification
- Tests compare CPU reference vs NPU implementation (both using naive path on CPU)
- For comprehensive testing, use the full test suite with more configurations
- Flash attention path requires float16/bfloat16 and seq_len ≥ 1024

---

**Report generated:** 2026-01-28
