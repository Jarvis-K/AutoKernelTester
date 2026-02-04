# Claude Skills for AutoKernelTester

本目录包含一套用于 **PyTorch 算子精度测试** 的 Claude 技能，帮助你自动化地对算子进行重构、测试规划和执行验证。

---

## 🎯 核心功能

这套技能可以帮助你：

1. **将复杂算子文件拆分为模块化包结构**
2. **自动推断 prefill/decode 测试场景**
3. **执行 Golden 对照测试并生成详细报告**
4. **在每个阶段与用户交互确认，确保测试计划的准确性**

---

## 📁 技能目录

```
.claude/skills/
├── test-op/                     # 🎮 主入口技能
│   └── SKILL.md
├── restructure-operator/        # 🔧 重构技能
│   ├── SKILL.md
│   └── reference/               # 代码模板
├── plan-operator-test/          # 📋 规划技能
│   └── SKILL.md
└── execute-operator-test/       # 🚀 执行技能
    └── SKILL.md
```

---

## 🚀 快速开始

### 使用主入口技能

```
/test-op <算子文件路径>
```

这会触发完整的三阶段流程：

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  阶段 1：重构    │  →   │  阶段 2：规划    │  →   │  阶段 3：执行    │
│  restructure    │      │  plan           │      │  execute        │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        ↓                        ↓                        ↓
   模块化包结构              测试计划文档              测试报告 + CSV
```

### 单独使用子技能

你也可以单独调用每个阶段的技能：

| 命令 | 说明 |
|------|------|
| `/restructure-operator` | 只执行重构阶段 |
| `/plan-operator-test` | 只执行规划阶段（需先完成重构） |
| `/execute-operator-test` | 只执行测试阶段（需先完成规划） |

---

## 📖 技能详解

### 1️⃣ test-op（主入口）

**用途**：算子精度测试的统一入口，自动路由到三个子技能。

**特点**：
- 强制按顺序执行：重构 → 规划 → 执行
- 每个阶段结束后暂停，等待用户确认
- 不确定的点会主动询问用户
- 用户反馈会沉淀到产物中

---

### 2️⃣ restructure-operator（重构）

**用途**：将单一复杂算子文件拆分为模块化包。

**输出结构**：
```
op_<算子名>/
├── __init__.py          # 薄导出
├── original.py          # 原始文件保留
├── utils.py             # 常量、辅助函数
├── cpu.py               # CPU 实现
├── npu.py               # NPU 实现（含 wrapper）
├── api.py               # 入口调度
├── test_cases.py        # 测试用例数据生成
├── test.py              # CLI 入口
├── testing/             # Golden 测试框架
└── tests/               # pytest 入口
```

**关键特性**：
- 使用 **复制-编辑** 方法，保留原始文件
- **Golden 对照验证**：确保重构后行为与原始一致
- **自动修复循环**：语法错误、导入错误会自动尝试修复（最多 5 次）
- 导出 CSV 格式的校验结果

---

### 3️⃣ plan-operator-test（规划）

**用途**：基于参数说明自动推断测试计划。

**自动推断**：
- **执行模式**：检测 `start_pos`、`cache`、`kv` → decode 模式；检测 `seq_len`、`cu_seqlens` → prefill 模式
- **形状覆盖**：自动识别 batch_size、seq_len、head_dim 等维度
- **分支参数**：bool 类型覆盖 True/False，enum 类型覆盖典型值

**输出产物**：
- `logs/test_op_plan.md` - 详细测试计划

---

### 4️⃣ execute-operator-test（执行）

**用途**：根据测试计划生成用例、运行测试、输出报告。

**执行命令**：
```bash
python -m op_<算子名>.test
```

**输出产物**：
- `op_<算子名>/test_cases.py` - 测试用例
- `logs/test_op_run.log` - 运行日志
- `logs/test_op_report.md` - 中文报告
- `golden_results_<timestamp>.csv` - 详细结果

**失败分类**：
| 类型 | 说明 |
|-----|------|
| `import_fail` | 导入失败 |
| `baseline_error` | 原始算子报错 |
| `numeric_mismatch` | 数值差异 |
| `shape_mismatch` | 形状不匹配 |
| `dtype_mismatch` | 类型不匹配 |

---

## ⚠️ 重要注意事项

> [!CAUTION]
> **必须前台交互式执行 - 禁止后台运行**
>
> 执行此技能时必须：
> 1. 在前台逐步执行每个阶段
> 2. 每个阶段完成后**停止并等待用户确认**
> 3. 所有命令输出对用户可见
> 4. 用户可随时中断或提供反馈

---

## 💬 用户交互示例

### 阶段 1 完成后
```
✅ 重构完成。请查看：
- 模块化包：op_<算子名>/
- 校验 CSV：golden_results_xxx.csv

请回复：
- "继续" - 进入规划阶段
- "调整 xxx" - 我会根据反馈修改并更新
```

### 阶段 2 完成后
```
✅ 测试计划已生成：logs/test_op_plan.md

请回复：
- "批准" - 开始执行测试
- "调整 seq_len 到 512" - 我会更新计划
```

### 阶段 3 完成后
```
✅ 测试完成：N/M 通过

请回复：
- "完成" - 结束测试流程
- "重跑失败用例" - 只重跑失败的
```

---

## 📂 产物汇总

| 阶段 | 产物 | 说明 |
|-----|------|------|
| 重构 | `op_<算子名>/` | 模块化包 |
| 重构 | `golden_results_xxx.csv` | 重构校验结果 |
| 规划 | `logs/test_op_plan.md` | 测试计划 |
| 执行 | `logs/test_op_run.log` | 运行日志 |
| 执行 | `logs/test_op_report.md` | 测试报告 |
| 执行 | `golden_results_<ts>.csv` | 测试详细结果 |

---

## 🔧 容差策略

| dtype | atol | rtol |
|-------|------|------|
| fp32 | 1e-6 | 1e-6 |
| fp16 | 1e-2 | 1e-2 |
| bf16 | 2e-2 | 2e-2 |

> 涉及 softmax/norm 的操作会自动放宽容差。

---

## 📝 反馈沉淀机制

用户的每次反馈都会被记录并更新到对应产物中：

| 用户反馈 | 沉淀位置 |
|---------|---------| 
| 入口映射调整 | `test_cases.py` 中的 entry 配置 |
| 形状建议 | `logs/test_op_plan.md` 更新覆盖轴 |
| 容差调整 | `test_cases.py` 中的 tols 配置 |
| 场景补充 | `test_cases.py` 添加新 case |
