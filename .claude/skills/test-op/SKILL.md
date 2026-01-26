---
name: test-op
description: Test PyTorch operator precision - routes to restructure, analyze, plan, and execute skills
---

# Test Operator Precision

> [!CAUTION]
> **必须前台交互式执行 - 禁止后台运行**
> 
> 执行此 skill 时，**禁止使用后台任务或异步执行**。必须：
> 1. 在前台逐步执行每个阶段
> 2. 每个阶段完成后**必须停止并等待用户确认**
> 3. 所有命令输出必须对用户可见
> 4. 用户必须能够随时中断或提供反馈
> 
> **原因**：此 skill 是交互式工作流，需要用户在每个阶段审核和确认才能继续。

This is the **main entry point** for testing PyTorch operator precision. It orchestrates up to four phases by routing to specialized skills.

> [!IMPORTANT]
> **Workflow Overview**
> 
> This skill coordinates up to four phases:
> 0. **Restructure** (optional, for complex operators) → Split into modular files
> 1. **Analyze** → `/analyze-operator` 
> 2. **Plan** → `/plan-operator-test`
> 3. **Execute** → `/execute-operator-test`
> 
> Each phase requires user confirmation before proceeding to the next.

---

## Complex Operator Detection

> [!WARNING]
> **Complex Operator Criteria**
> 
> An operator is considered **complex** and requires restructuring if ANY of these conditions are met:
> 
> | Criteria | Threshold |
> |----------|-----------|
> | Total lines of code | > 300 lines |
> | Number of input tensors | ≥ 5 inputs |
> | Number of parameters | ≥ 8 parameters |
> | Number of utility/helper functions | ≥ 3 functions |
> | Nested function definitions | Any nested functions |
> | Multiple algorithm variants | > 1 code path |
> | External dependencies | Custom utils from other files |

When an operator meets any of these criteria, the **Restructure Phase** is mandatory before analysis.

---

## Quick Start

When user invokes `/test-op [file]`:

### If operator is complex (no prior restructuring):
→ Execute Restructure Phase first

### If restructured (or operator is simple) and no prior analysis:
→ Route to `/analyze-operator` with the file path

### If user just confirmed analysis:
→ Route to `/plan-operator-test`

### If user just approved test plan:
→ Route to `/execute-operator-test`

---

## Phase Routing Logic

```
User: /test-op sample_operator.py
  ↓
Check: Is operator complex?
  ├─ YES → Check: Has restructuring been done?
  │         ├─ NO  → Execute Restructure Phase
  │         │        → Split into modular files
  │         │        → Verify equivalence to original
  │         │        → Ask user to confirm restructured code
  │         │        → STOP
  │         │
  │         └─ YES → Continue to Analysis
  │
  └─ NO (simple operator) → Continue to Analysis
       ↓
Check: Has analysis been done?
  ├─ NO  → Execute /analyze-operator instructions
  │        → Generate analysis report
  │        → Ask user to confirm
  │        → STOP
  │
  └─ YES (user confirmed analysis)
       ↓
     Check: Has test plan been approved?
       ├─ NO  → Execute /plan-operator-test instructions
       │        → Generate test plan
       │        → Ask user to approve
       │        → STOP
       │
       └─ YES (user approved plan)
            ↓
          Execute /execute-operator-test instructions
            → Run tests
            → Export CSV
            → Generate report
            → Present results
```

---

## How to Determine Current Phase

0. **Complex operator without restructured files** → Start Phase 0 (restructure)
1. **No analysis report exists** → Start Phase 1 (analyze)
2. **Analysis confirmed but no test plan** → Start Phase 2 (plan)
3. **Test plan approved** → Start Phase 3 (execute)

When in doubt, ask the user which phase to start from.

---

## Instructions for Each Phase

### Phase 0: Restructure (for complex operators only)

> [!NOTE]
> **Goal**: Break down a complex operator into modular, testable components while maintaining equivalence to the original implementation.

**Steps:**

#### 0.1 Complexity Assessment
1. Read the operator file completely
2. Count lines, inputs, parameters, and helper functions
3. Determine if restructuring is needed using the criteria above
4. If not complex, skip to Phase 1

#### 0.2 Identify Components
Categorize code into:
| Component Type | Description | File Suffix |
|----------------|-------------|-------------|
| Core Logic | Main operator computation | `_core.py` |
| Input Validation | Type checking, shape validation | `_validation.py` |
| Utility Functions | Helper math functions, data manipulation | `_utils.py` |
| CPU Reference | Reference implementation | `_cpu.py` |
| NPU/GPU Kernel | Accelerator implementation | `_npu.py` or `_cuda.py` |
| Constants/Config | Thresholds, default values | `_config.py` |

#### 0.3 Create Restructured Files
Create a new directory: `[operator_name]_structured/`

Structure:
```
[operator_name]_structured/
├── __init__.py          # Re-export main functions
├── [op]_core.py         # Core computation logic
├── [op]_utils.py        # Utility functions
├── [op]_validation.py   # Input validation
├── [op]_cpu.py          # CPU reference implementation
├── [op]_npu.py          # NPU implementation
├── [op]_config.py       # Constants and configuration
└── [op]_original.py     # Copy of original file for reference
```

#### 0.4 Verify Equivalence

> [!CAUTION]
> **CRITICAL**: The restructured code MUST produce identical outputs to the original.

Create a verification script that:
1. Imports both original and restructured versions
2. Tests with multiple input configurations
3. Uses `torch.allclose()` to verify numerical equivalence
4. Reports any discrepancies

```python
# Example verification
def verify_equivalence(original_fn, restructured_fn, test_inputs):
    for inputs in test_inputs:
        orig_output = original_fn(*inputs)
        new_output = restructured_fn(*inputs)
        assert torch.allclose(orig_output, new_output, rtol=1e-7, atol=1e-7), \
            f"Output mismatch for inputs {inputs}"
    print("✅ Equivalence verified!")
```

#### 0.5 Generate Restructure Report

Create a markdown report:

```markdown
# Restructure Report: [OPERATOR_NAME]

## Complexity Assessment
| Metric | Value | Threshold | Complex? |
|--------|-------|-----------|----------|
| Lines of code | X | > 300 | Yes/No |
| Input tensors | X | ≥ 5 | Yes/No |
| Parameters | X | ≥ 8 | Yes/No |
| Helper functions | X | ≥ 3 | Yes/No |

## Created Files
| File | Purpose | Lines |
|------|---------|-------|
| `_core.py` | Main computation | X |
| `_utils.py` | Utilities | X |
| ... | ... | ... |

## Equivalence Verification
- Test cases run: N
- All passed: ✅/❌
- Numerical precision: rtol=X, atol=Y

## Original vs Restructured Mapping
| Original Location | New Location |
|-------------------|--------------|
| `def helper_fn()` (L10-25) | `_utils.py:helper_fn()` |
| `def cpu_impl()` (L30-100) | `_cpu.py:compute()` |
| ... | ... |
```

#### 0.6 Request User Confirmation

```
I've restructured the complex operator into modular files.

**Summary:**
- Original: 1 file, X lines
- Restructured: N files, better organized
- Equivalence: ✅ Verified with M test cases

Please review the restructured files and the equivalence report.

**Questions:**
1. Is the modular structure appropriate?
2. Should any components be combined or further split?

**Next step:** After confirming, we'll proceed with analysis using the restructured code.
```

**STOP and wait for user confirmation.**

---

### Phase 1: Analyze (if no analysis yet)

Read and follow: `.claude/skills/analyze-operator/SKILL.md`

Key steps:
1. Read the operator file (use restructured files if available)
2. Identify CPU/NPU functions
3. Classify parameters
4. Generate analysis report markdown
5. **STOP and ask user to confirm**

> [!TIP]
> For restructured operators, analyze each module separately and then synthesize into a single comprehensive report.

### Phase 2: Plan (after user confirms analysis)

Read and follow: `.claude/skills/plan-operator-test/SKILL.md`

Key steps:
1. Review confirmed analysis
2. Design test configurations
3. Set tolerance thresholds
4. Generate test plan markdown
5. **STOP and ask user to approve**

### Phase 3: Execute (after user approves plan)

Read and follow: `.claude/skills/execute-operator-test/SKILL.md`

Key steps:
1. Generate test script
2. Run tests
3. Export CSV results
4. Generate final report
5. Present results to user

---

## Example Conversation Flow

### Simple Operator (no restructuring needed)
```
User: /test-op simple_relu.py

Agent: [Generates analysis report]
       Please review the analysis. Is it correct?

User: Looks good, continue.

Agent: [Generates test plan]
       Please review the test plan. Do you approve?

User: Yes, run the tests.

Agent: [Runs tests, generates CSV and report]
       Tests complete! 45/50 passed. See report and CSV.
```

### Complex Operator (restructuring required)
```
User: /test-op complex_attention.py

Agent: This operator is complex (450 lines, 7 inputs, 12 parameters).
       I'll restructure it into modular files first.
       
       [Creates complex_attention_structured/ directory]
       [Generates restructure report]
       [Runs equivalence verification]
       
       Please review the restructured code and equivalence report.

User: The structure looks good, continue.

Agent: [Generates analysis report based on restructured code]
       Please review the analysis. Is it correct?

User: Yes, proceed.

Agent: [Generates test plan]
       Please review the test plan. Do you approve?

User: Approved.

Agent: [Runs tests, generates CSV and report]
       Tests complete! 120/128 passed. See report and CSV.
```

---

## Important Rules

1. **Assess complexity first** - Check if restructuring is needed before any other work
2. **Never skip phases** - Always go in order: (Restructure) → Analyze → Plan → Execute
3. **Always verify equivalence** - Restructured code must match original behavior exactly
4. **Always wait for user confirmation** between phases
5. **Generate markdown artifacts** for restructure report, analysis report, and test plan
6. **Export CSV** with test results in Execute phase
