---
name: execute-operator-test
description: Execute precision tests for an approved operator test plan and generate results
---

# Execute Operator Test

Execute precision tests based on an approved test plan and generate comprehensive results.

> [!IMPORTANT]
> This is **Step 3 of 3** in the operator testing workflow.
> 
> **Prerequisite**: User must have approved the test plan from `/plan-operator-test`

---

## Input

- Approved test plan from Step 2
- Original operator file path

---

## Steps

### 1. Generate Test Script

Create a self-contained Python test script:

```python
#!/usr/bin/env python3
"""Precision test for [OPERATOR_NAME]"""

import torch
import csv
import json
from typing import Dict, List, Any

# Import user's operator (adjust path as needed)
# from [module] import cpu_func, npu_func

# Test configurations (from approved plan)
SHAPES = [...]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
PATTERNS = ["random", "zeros", "very_small"]

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-8},
    torch.float16: {"rtol": 1e-3, "atol": 1e-4},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-3},
}

def generate_tensor(shape, dtype, pattern):
    """Generate test tensor with specified pattern."""
    if pattern == "random":
        return torch.randn(shape, dtype=dtype)
    elif pattern == "zeros":
        return torch.zeros(shape, dtype=dtype)
    elif pattern == "ones":
        return torch.ones(shape, dtype=dtype)
    elif pattern == "very_small":
        return torch.randn(shape, dtype=dtype) * 1e-7
    elif pattern == "very_large":
        return torch.randn(shape, dtype=dtype) * 1e5
    else:
        return torch.randn(shape, dtype=dtype)

def compute_metrics(ref, test):
    """Compute precision metrics."""
    ref = ref.float().cpu()
    test = test.float().cpu()
    diff = torch.abs(ref - test)
    denom = torch.abs(ref).clamp(min=1e-8)
    return {
        "max_abs_diff": diff.max().item(),
        "max_rel_diff": (diff / denom).max().item(),
        "mean_abs_diff": diff.mean().item(),
        "mse": torch.mean((ref - test) ** 2).item(),
    }

def run_tests():
    results = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "details": []}
    
    for shape in SHAPES:
        for dtype in DTYPES:
            for pattern in PATTERNS:
                results["total"] += 1
                config = {"shape": shape, "dtype": str(dtype), "pattern": pattern}
                
                try:
                    tensor = generate_tensor(shape, dtype, pattern)
                    # Run reference and test implementations
                    # ref_out = cpu_func(tensor)
                    # test_out = npu_func(tensor.npu()).cpu()
                    
                    # metrics = compute_metrics(ref_out, test_out)
                    # tol = TOLERANCES[dtype]
                    # passed = metrics["max_abs_diff"] <= tol["atol"] or metrics["max_rel_diff"] <= tol["rtol"]
                    
                    # Placeholder - replace with actual test
                    passed = True
                    metrics = {"max_abs_diff": 0, "max_rel_diff": 0}
                    
                    if passed:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                    
                    results["details"].append({
                        "config": config,
                        "passed": passed,
                        "metrics": metrics
                    })
                except Exception as e:
                    results["errors"] += 1
                    results["details"].append({
                        "config": config,
                        "passed": False,
                        "error": str(e)
                    })
    
    return results

if __name__ == "__main__":
    results = run_tests()
    print(f"Results: {results['passed']}/{results['total']} passed")
```

### 2. Run Tests

Execute the generated script:
```bash
python test_[operator_name].py
```

### 3. Export CSV Results

Generate a CSV file with all test results:

```python
def export_csv(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'test_id', 'shape', 'dtype', 'pattern', 
            'status', 'max_abs_diff', 'max_rel_diff', 'mse', 'error'
        ])
        writer.writeheader()
        for i, d in enumerate(results['details']):
            writer.writerow({
                'test_id': i + 1,
                'shape': str(d['config']['shape']),
                'dtype': d['config']['dtype'],
                'pattern': d['config']['pattern'],
                'status': 'PASS' if d['passed'] else ('ERROR' if 'error' in d else 'FAIL'),
                'max_abs_diff': d.get('metrics', {}).get('max_abs_diff', ''),
                'max_rel_diff': d.get('metrics', {}).get('max_rel_diff', ''),
                'mse': d.get('metrics', {}).get('mse', ''),
                'error': d.get('error', '')
            })
```

**CSV Format:**
```csv
test_id,shape,dtype,pattern,status,max_abs_diff,max_rel_diff,mse,error
1,"(1,3,32,32)",torch.float32,random,PASS,1.2e-7,3.4e-6,1.1e-14,
2,"(4,64,56,56)",torch.float16,zeros,PASS,0.0,0.0,0.0,
3,"(16,128,14,14)",torch.float32,very_large,FAIL,5.6e-3,2.1e-2,3.2e-5,
```

### 4. Generate Final Report

Create a comprehensive markdown report:

```markdown
# Precision Test Report: [OPERATOR_NAME]

## Summary
| Metric | Value |
|--------|-------|
| Total Tests | N |
| Passed | X |
| Failed | Y |
| Errors | Z |
| Pass Rate | P% |
| CSV Export | `[operator]_results.csv` |

## Results by Dtype
| Dtype | Passed | Failed | Pass Rate |
|-------|--------|--------|-----------|
| float32 | X | Y | P% |
| float16 | X | Y | P% |
| bfloat16 | X | Y | P% |

## Failed Tests
[List of failed test configurations with metrics]

## Recommendations
[Based on results]
```

### 5. Present Results

Use `notify_user` to present:
- Test summary
- Link to full report
- Link to CSV file
- Any recommendations

---

## Output

- Test script file
- CSV results file
- Final report markdown
- Summary to user
