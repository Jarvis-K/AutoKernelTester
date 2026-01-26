---
name: test-op
description: Test PyTorch operator precision - routes to analyze, plan, and execute skills
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

This is the **main entry point** for testing PyTorch operator precision. It orchestrates three phases by routing to specialized skills.

> [!IMPORTANT]
> **Workflow Overview**
> 
> This skill coordinates three phases:
> 1. **Analyze** → `/analyze-operator` 
> 2. **Plan** → `/plan-operator-test`
> 3. **Execute** → `/execute-operator-test`
> 
> Each phase requires user confirmation before proceeding to the next.

---

## Quick Start

When user invokes `/test-op [file]`:

### If starting fresh (no prior analysis):
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

1. **No analysis report exists** → Start Phase 1 (analyze)
2. **Analysis confirmed but no test plan** → Start Phase 2 (plan)
3. **Test plan approved** → Start Phase 3 (execute)

When in doubt, ask the user which phase to start from.

---

## Instructions for Each Phase

### Phase 1: Analyze (if no analysis yet)

Read and follow: `.claude/skills/analyze-operator/SKILL.md`

Key steps:
1. Read the operator file
2. Identify CPU/NPU functions
3. Classify parameters
4. Generate analysis report markdown
5. **STOP and ask user to confirm**

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

```
User: /test-op sample_operator.py

Agent: [Generates analysis report]
       Please review the analysis. Is it correct?

User: Looks good, continue.

Agent: [Generates test plan]
       Please review the test plan. Do you approve?

User: Yes, run the tests.

Agent: [Runs tests, generates CSV and report]
       Tests complete! 45/50 passed. See report and CSV.
```

---

## Important Rules

1. **Never skip phases** - Always go in order: Analyze → Plan → Execute
2. **Always wait for user confirmation** between phases
3. **Generate markdown artifacts** for analysis report and test plan
4. **Export CSV** with test results in Execute phase
