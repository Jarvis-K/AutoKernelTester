---
name: restructure-operator
description: Restructure complex Python operators into modular files with incremental verification
---

# Restructure Operator

> [!CAUTION]
> **必须前台交互式执行 - 禁止后台运行**
> 
> 执行此 skill 时，**禁止使用后台任务或异步执行**。必须：
> 1. 在前台逐步执行每个步骤
> 2. 每个组件提取后**必须停止并等待验证通过**
> 3. 所有输出必须对用户可见

Restructure a complex Python operator file into modular, testable components using an **incremental approach** for higher success rates.

> [!IMPORTANT]
> **Key Principle: Incremental Extraction**
> 
> Do NOT attempt to split the entire file at once. Instead:
> 1. Extract ONE component at a time
> 2. Verify equivalence after EACH extraction
> 3. Only proceed to next component after verification passes

---

## When to Use This Skill

An operator requires restructuring if **ANY** of these conditions are met:

| Criteria | Threshold |
|----------|-----------|
| Total lines of code | > 300 lines |
| Number of input tensors | ≥ 5 inputs |
| Number of parameters | ≥ 8 parameters |
| Number of utility/helper functions | ≥ 3 functions |
| Nested function definitions | Any nested functions |
| Multiple algorithm variants | > 1 code path |

---

## Restructuring Process

### Step 1: Full File Analysis

**Before any modifications:**

1. Read the entire file with `view_file`
2. Use `view_file_outline` to get the AST structure
3. Create a dependency graph of all functions:

```markdown
## Function Dependency Analysis

| Function | Line Range | Depends On | Called By |
|----------|------------|------------|-----------|
| `helper_a()` | L10-25 | (none) | `cpu_impl` |
| `helper_b()` | L30-50 | `helper_a` | `cpu_impl`, `npu_impl` |
| `cpu_impl()` | L55-150 | `helper_a`, `helper_b` | (entry point) |
| `npu_impl()` | L155-280 | `helper_b` | (entry point) |
```

4. Identify extraction order (bottom-up by dependency):
   - **First**: Functions with NO dependencies (leaf nodes)
   - **Next**: Functions depending only on already-extracted functions
   - **Last**: Entry point functions

---

### Step 2: Create Directory Structure

Create the restructured directory:

```
[operator_name]_structured/
├── __init__.py          # Re-export main functions (create LAST)
├── [op]_config.py       # Constants and configuration (extract FIRST)
├── [op]_utils.py        # Utility functions (extract SECOND)
├── [op]_validation.py   # Input validation (extract THIRD)
├── [op]_cpu.py          # CPU reference implementation
├── [op]_npu.py          # NPU implementation
└── [op]_original.py     # Copy of original file for reference
```

**First action**: Copy the original file to `[op]_original.py` as reference.

---

### Step 3: Incremental Extraction

> [!WARNING]
> **Extract ONE component at a time. Verify after EACH extraction.**

#### 3.1 Extract Constants/Config (Safest First)

1. Find all module-level constants, type aliases, default values
2. Create `[op]_config.py` with these items
3. In original file: replace with `from .[op]_config import ...`
4. **VERIFY**: Run original tests if available, or import check

#### 3.2 Extract Utility Functions (Leaf Nodes)

For EACH utility function (in dependency order):

1. Identify a single function with no internal dependencies
2. Move to `[op]_utils.py`
3. Add import to original file
4. **VERIFY**: 
   ```python
   # Quick verification
   from [original] import target_function
   from [op]_structured import target_function as new_fn
   # Run test case
   assert torch.allclose(target_function(x), new_fn(x))
   ```

5. Only proceed to next function after verification passes

#### 3.3 Extract Validation Logic

1. Find input validation code (type checks, shape assertions)
2. Move to `[op]_validation.py`
3. **VERIFY**: Import check and basic call test

#### 3.4 Extract CPU Implementation

1. Move CPU reference implementation to `[op]_cpu.py`
2. Update imports
3. **VERIFY**: Run with test inputs

#### 3.5 Extract NPU/GPU Implementation

1. Move accelerator implementation to `[op]_npu.py`
2. Update imports
3. **VERIFY**: Run with test inputs

---

### Step 4: Create `__init__.py`

Only after ALL components are extracted and verified:

```python
# [op]_structured/__init__.py
"""Restructured [OPERATOR_NAME] operator."""

from .[op]_cpu import cpu_impl_function
from .[op]_npu import npu_impl_function

# Re-export main entry points
__all__ = ['cpu_impl_function', 'npu_impl_function']
```

---

### Step 5: Final Equivalence Verification

Run comprehensive verification:

```python
import torch
from [original_module] import original_cpu_fn, original_npu_fn
from [op]_structured import cpu_impl_function, npu_impl_function

def verify_full_equivalence():
    test_shapes = [(1, 64), (4, 128), (2, 3, 256)]
    test_dtypes = [torch.float32, torch.float16]
    
    for shape in test_shapes:
        for dtype in test_dtypes:
            x = torch.randn(shape, dtype=dtype)
            
            # CPU verification
            orig_cpu = original_cpu_fn(x)
            new_cpu = cpu_impl_function(x)
            assert torch.allclose(orig_cpu, new_cpu, rtol=1e-7, atol=1e-7), \
                f"CPU mismatch for shape={shape}, dtype={dtype}"
            
            # NPU verification (if applicable)
            if torch.npu.is_available():
                x_npu = x.npu()
                orig_npu = original_npu_fn(x_npu).cpu()
                new_npu = npu_impl_function(x_npu).cpu()
                assert torch.allclose(orig_npu, new_npu, rtol=1e-7, atol=1e-7), \
                    f"NPU mismatch for shape={shape}, dtype={dtype}"
    
    print("✅ Full equivalence verified!")

verify_full_equivalence()
```

---

### Step 6: Generate Restructure Report

Create a markdown report summarizing the restructuring:

```markdown
# Restructure Report: [OPERATOR_NAME]

## Original File
- **Path**: `[original_path]`
- **Lines**: X
- **Functions**: Y

## Extraction Order
| Step | Component | File | Verification |
|------|-----------|------|--------------|
| 1 | Constants | `_config.py` | ✅ |
| 2 | `helper_a()` | `_utils.py` | ✅ |
| 3 | `helper_b()` | `_utils.py` | ✅ |
| 4 | Validation | `_validation.py` | ✅ |
| 5 | CPU impl | `_cpu.py` | ✅ |
| 6 | NPU impl | `_npu.py` | ✅ |

## Created Files
| File | Purpose | Lines |
|------|---------|-------|
| `_config.py` | Constants/defaults | X |
| `_utils.py` | Helper functions | X |
| `_validation.py` | Input validation | X |
| `_cpu.py` | CPU implementation | X |
| `_npu.py` | NPU implementation | X |
| `__init__.py` | Module exports | X |
| `_original.py` | Reference copy | X |

## Mapping: Original → Restructured
| Original Location | New Location |
|-------------------|--------------|
| `CONST_A` (L5) | `_config.py:CONST_A` |
| `helper_a()` (L10-25) | `_utils.py:helper_a()` |
| `cpu_impl()` (L55-150) | `_cpu.py:compute()` |

## Final Verification
- Test configurations: X
- All passed: ✅
- Numerical precision: rtol=1e-7, atol=1e-7
```

---

### Step 7: Request User Confirmation

```
I've restructured the complex operator into modular files using incremental extraction.

**Summary:**
- Original: 1 file, X lines, Y functions
- Restructured: N files, modular organization
- Extraction steps: M (all verified ✅)

Please review the restructured files and the restructure report.

**Questions:**
1. Is the modular structure appropriate?
2. Should any components be combined or further split?

**Next step:** After confirming, use `/test-op` to proceed with analysis and testing.
```

**STOP and wait for user confirmation.**

---

## Error Recovery

If verification fails at any step:

1. **DO NOT PROCEED** to the next extraction
2. Revert the last change
3. Analyze why the extraction broke equivalence
4. Try alternative extraction approach:
   - Keep more context in the extracted function
   - Check for hidden state dependencies
   - Verify import order

---

## Output

- Restructured directory with modular files
- Original file preserved as `_original.py`
- Restructure report markdown
- User confirmation request
