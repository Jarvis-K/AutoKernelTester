---
name: restructure-operator
description: Restructure complex Python operators using copy-and-edit for speed
---

# Restructure Operator

> [!CAUTION]
> **禁止后台运行** - 必须前台逐步执行，每步等待验证。

Split complex operator files using a **copy-and-edit** approach for faster, safer refactoring.

> [!IMPORTANT]
> **Copy-and-Edit Principle**
> 
> 1. **Copy** the original file as the base for each new module
> 2. **Delete** irrelevant code from each copy
> 3. **Add** imports to connect modules
> 
> This is FASTER and SAFER than generating code from scratch.

---

## Complexity Thresholds

Restructure if ANY condition is met:

| Criteria | Threshold |
|----------|-----------|
| Lines of code | > 300 |
| Input tensors | ≥ 5 |
| Parameters | ≥ 8 |
| Helper functions | ≥ 3 |

---

## Process

### Step 1: Analyze (Quick)

1. `view_file_outline` to list all functions
2. Identify: constants, utils, CPU impl, NPU impl
3. Note line ranges for each component

### Step 2: Setup Directory

```bash
mkdir [op]_structured
cp original.py [op]_structured/[op]_original.py  # Keep reference
```

### Step 3: Copy-and-Edit Each Module

> [!TIP]
> **For each module**: Copy original → Delete unneeded code → Add imports

#### 3.1 Utils Module

```bash
cp original.py [op]_structured/[op]_utils.py
```

Then **edit** `[op]_utils.py`:
- Keep only: helper functions, constants they need
- Delete: CPU/NPU implementations, main logic

#### 3.2 CPU Module

```bash
cp original.py [op]_structured/[op]_cpu.py
```

Then **edit** `[op]_cpu.py`:
- Keep only: CPU implementation function
- Add: `from .[op]_utils import ...`
- Delete: NPU code, unused helpers

#### 3.3 NPU Module

```bash
cp original.py [op]_structured/[op]_npu.py
```

Then **edit** `[op]_npu.py`:
- Keep only: NPU implementation function
- Add: `from .[op]_utils import ...`
- Delete: CPU code, unused helpers

#### 3.4 Init Module

Create `__init__.py`:
```python
from .[op]_cpu import cpu_fn
from .[op]_npu import npu_fn
__all__ = ['cpu_fn', 'npu_fn']
```

### Step 4: Quick Verification

```python
# Import test
from [op]_structured import cpu_fn, npu_fn
x = torch.randn(2, 64)
cpu_fn(x)  # Should not error
```

### Step 5: Report to User

```
Restructured [OP] using copy-and-edit:
- [op]_utils.py: X lines (helpers)
- [op]_cpu.py: Y lines (CPU impl)
- [op]_npu.py: Z lines (NPU impl)
- Original preserved in _original.py

Ready for /analyze-operator
```

**STOP and wait for confirmation.**

---

## Output

- `[op]_structured/` directory with modular files
- Original preserved as `_original.py`
