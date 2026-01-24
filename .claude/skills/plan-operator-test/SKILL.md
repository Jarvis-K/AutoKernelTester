---
name: plan-operator-test
description: Generate a test plan for an analyzed PyTorch operator
---

# Plan Operator Test

Generate a comprehensive test plan for an operator that has been analyzed.

> [!IMPORTANT]
> This is **Step 2 of 3** in the operator testing workflow.
> 
> **Prerequisite**: User must have confirmed the analysis from `/analyze-operator`
> 
> After this skill completes, tell the user:
> **"Use `/execute-operator-test` to run the tests after approving the plan."**

---

## Input

- Confirmed analysis report from Step 1
- Any user feedback or parameter specifications

---

## Steps

### 1. Review Analysis

Check what operator type and parameters were identified in the analysis report.

### 2. Select Test Configurations

Based on operator type, select appropriate test configurations:

#### Input Tensor Shapes

| Operator Type | Recommended Shapes |
|---------------|-------------------|
| elementwise | `(1,), (32,), (4, 64), (2, 3, 224, 224)` |
| reduction | `(1024,), (4, 256), (2, 32, 128)` |
| matmul | `(1, 512, 4096), (2, 1024, 4096), (1, 2048, 4096)` |
| norm | `(1, 512, 768), (1, 2048, 4096), (2, 1024, 4096)` |
| attention | `(1, 32, 512, 128), (1, 32, 2048, 128)` |
| conv | `(1, 3, 32, 32), (4, 64, 56, 56), (2, 128, 14, 14)` |

#### Batch Size Prior
```python
# Always test these batch sizes for thorough coverage
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48]
```

> [!TIP]
> When generating test shapes, vary the batch dimension using the above values.
> For example, for a norm operator with shape `(B, S, H)`:
> - `(1, 512, 768)`, `(2, 512, 768)`, `(4, 512, 768)`, ..., `(48, 512, 768)`

#### Data Types
```python
dtypes = [torch.float32, torch.float16, torch.bfloat16]
```

#### Value Patterns
```python
patterns = ["random", "zeros", "ones", "very_small", "very_large", "mixed_sign"]
```

### 3. Set Tolerance Thresholds

| Operator Type | float32 rtol/atol | float16 rtol/atol | bfloat16 rtol/atol |
|--------------|-------------------|-------------------|---------------------|
| Elementwise | 1e-5 / 1e-8 | 1e-3 / 1e-4 | 1e-2 / 1e-3 |
| Reduction | 1e-4 / 1e-6 | 1e-3 / 1e-4 | 1e-2 / 1e-3 |
| MatMul | 1e-4 / 1e-6 | 1e-3 / 1e-4 | 1e-2 / 1e-3 |
| Normalization | 1e-3 / 1e-5 | 1e-2 / 1e-4 | 5e-2 / 1e-3 |
| Attention | 1e-3 / 1e-4 | 1e-2 / 1e-3 | 5e-2 / 1e-2 |

### 4. Generate Test Plan

Create a markdown artifact:

```markdown
# Test Plan: [OPERATOR_NAME]

## Test Summary
| Category | Count |
|----------|-------|
| Shapes | X |
| Dtypes | Y |
| Patterns | Z |
| **Total Tests** | **N** |

## Test Configurations

### Input Tensor: `[param_name]`
| Shapes | Dtypes | Patterns |
|--------|--------|----------|
| `[(1,3,32,32), ...]` | `[float32, float16]` | `[random, zeros]` |

### Algorithm Parameters
| Parameter | Test Values |
|-----------|-------------|
| eps | `[1e-5, 1e-6, 1e-8]` |

## Tolerance Settings
| Dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-5 | 1e-8 |
| float16 | 1e-3 | 1e-4 |
| bfloat16 | 1e-2 | 1e-3 |

## Edge Cases
- [ ] Single element tensor
- [ ] Non-contiguous tensor
- [ ] Very large values
- [ ] Very small values

## Estimated Runtime
~N test cases, approximately M seconds
```

### 5. Request User Approval

Use `notify_user` with:

```
I've generated the test plan. Please review:

1. Are the test shapes appropriate?
2. Should any test cases be added or removed?
3. Are the tolerance settings correct?

**Next step:** After approval, use `/execute-operator-test` to run the tests.
```

---

## Output

- Test plan markdown file
- User approval request

---

## LLM Operator Shape Reference

For LLM operators, use these shape guidelines:

| Model Size | batch | seq_len | hidden | heads | head_dim |
|------------|-------|---------|--------|-------|----------|
| Small (125M) | 1-8 | 128-512 | 768 | 12 | 64 |
| Medium (1-3B) | 1-4 | 512-2048 | 2048 | 16-20 | 128 |
| Large (7-13B) | 1-2 | 1024-4096 | 4096 | 32-40 | 128 |
