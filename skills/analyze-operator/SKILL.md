---
name: analyze-operator
description: Analyze a PyTorch operator file and generate an analysis report for user review
---

# Analyze Operator

Analyze a PyTorch operator file to understand its structure, parameters, and testing requirements.

> [!IMPORTANT]
> This is **Step 1 of 3** in the operator testing workflow.
> 
> After this skill completes, tell the user:
> **"Use `/plan-operator-test` to generate a test plan after reviewing the analysis."**

---

## Input

User provides a Python file containing:
- CPU reference implementation
- NPU/GPU implementation to test
- (Optional) Simple test code showing usage

---

## Steps

### 1. Read the Source File

Use `view_file` to read the user's Python file.

### 2. Identify Functions

Find:
- **CPU function**: Usually named `*_cpu`, `*_reference`, or is the pure Python implementation
- **NPU/GPU function**: Usually named `*_npu`, `*_cuda`, or calls accelerator APIs like `torch_npu.*`

### 3. Analyze Parameters

For each parameter, classify using this table:

| Category | Identification | Test Priority |
|----------|----------------|---------------|
| Input Tensor | Type `Tensor`, used in computation | HIGH |
| Shape Parameter | Names like `kernel_size`, `stride`, `padding` | HIGH |
| Algorithm Parameter | Names like `eps`, `threshold`, `reduction` | MEDIUM |
| Control Flag | Boolean, `training`, `inplace` | LOW |
| Fixed/Internal | `seed`, `generator` | SKIP |

### 4. Infer Operator Type

Classify as one of:
- `elementwise` - Element-by-element operations (relu, sigmoid, add)
- `reduction` - Sum, mean, max, etc.
- `matmul` - Matrix multiplication, linear layers
- `conv` - Convolution operations
- `norm` - LayerNorm, BatchNorm, RMSNorm
- `attention` - Self-attention, cross-attention
- `embedding` - Token/position embeddings
- `other` - Anything else

### 5. Generate Analysis Report

Create a markdown artifact with this structure:

```markdown
# Operator Analysis Report: [OPERATOR_NAME]

## Source File
- **Path**: `[file_path]`
- **CPU Function**: `[function_name]`
- **NPU Function**: `[function_name]`

## Function Signature
\`\`\`python
def operator_name(param1: type, param2: type = default, ...) -> return_type
\`\`\`

## Parameter Analysis

| # | Parameter | Type | Default | Category | Test Priority |
|---|-----------|------|---------|----------|---------------|
| 1 | x | Tensor | required | Input Tensor | HIGH |
| 2 | weight | Tensor | required | Input Tensor | HIGH |
| 3 | eps | float | 1e-5 | Algorithm | MEDIUM |

## Inferred Operator Type
- **Type**: [operator_type]
- **Reasoning**: [brief explanation]

## Observations
- Input shapes from test code: [if present]
- Data types used: [if present]
- Special considerations: [any notes]

## Questions for User (if any)
1. [Any unclear aspects]
```

### 6. Request User Confirmation

Use `notify_user` with:

```
I've analyzed your operator file. Please review the analysis report.

**Questions:**
1. Is the parameter classification correct?
2. Would you like to provide specific test values for any parameters?

**Next step:** After confirming, use `/plan-operator-test` to generate a test plan.
```

---

## Output

- Analysis report markdown file
- User confirmation request
