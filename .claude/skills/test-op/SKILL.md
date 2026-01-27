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
> 0. **Restructure** (optional, for complex operators) → `/restructure-operator`
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
→ Route to `/restructure-operator` with the file path

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
  │         ├─ NO  → Route to /restructure-operator
  │         │        → STOP (skill handles user confirmation)
  │         │
  │         └─ YES → Continue to Analysis
  │
  └─ NO (simple operator) → Continue to Analysis
       ↓
Check: Has analysis been done?
  ├─ NO  → Route to /analyze-operator
  │        → STOP (skill handles user confirmation)
  │
  └─ YES (user confirmed analysis)
       ↓
     Check: Has test plan been approved?
       ├─ NO  → Route to /plan-operator-test
       │        → STOP (skill handles user confirmation)
       │
       └─ YES (user approved plan)
            ↓
          Route to /execute-operator-test
            → Run tests
            → Export CSV
            → Generate report
            → Present results
```

---

## How to Determine Current Phase

0. **Complex operator without restructured files** → Route to `/restructure-operator`
1. **No analysis report exists** → Route to `/analyze-operator`
2. **Analysis confirmed but no test plan** → Route to `/plan-operator-test`
3. **Test plan approved** → Route to `/execute-operator-test`

When in doubt, ask the user which phase to start from.

---

## Instructions for Each Phase

### Phase 0: Restructure (for complex operators only)

Read and follow: `.claude/skills/restructure-operator/SKILL.md`

> [!TIP]
> The restructure skill uses **incremental extraction** for higher success rates:
> - Extracts ONE component at a time
> - Verifies equivalence after EACH extraction
> - Only proceeds after verification passes

Key outputs:
- Restructured directory with modular files
- Restructure report with extraction steps
- Equivalence verification results

**After restructuring is confirmed, continue to Phase 1.**

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

---

### Phase 2: Plan (after user confirms analysis)

Read and follow: `.claude/skills/plan-operator-test/SKILL.md`

Key steps:
1. Review confirmed analysis
2. Design test configurations
3. Set tolerance thresholds
4. Generate test plan markdown
5. **STOP and ask user to approve**

---

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
       I'll use /restructure-operator to split it into modular files.
       
       [Routes to /restructure-operator]
       [Creates complex_attention_structured/ directory incrementally]
       [Verifies each extraction step]
       [Generates restructure report]
       
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
3. **Route to specialized skills** - Each phase has a dedicated skill with detailed instructions
4. **Always wait for user confirmation** between phases
5. **Generate markdown artifacts** for restructure report, analysis report, and test plan
6. **Export CSV** with test results in Execute phase
