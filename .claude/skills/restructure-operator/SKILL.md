---
name: restructure-operator
description: 使用复制-编辑方法重构复杂 Python 算子（模块化包 + Golden 测试框架 + 自动验证修复）
---

# 操作符重构

使用 **复制-编辑** 方法拆分复杂算子文件，通过 **Golden 对照验证** 保证重构后行为与原始一致。

---

## 模块化原则

> [!IMPORTANT]
> 所有模块采用 **包内绝对导入**：
> - ✅ `from op_<opname>.utils import foo`
> - ❌ `from .utils import foo`
>
> 运行测试：`python -m op_<opname>.test`

---

## 目标输出结构

```
op_<opname>/
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

---

## 执行流程

| 步骤 | 说明 |
|------|------|
| 0 | 准备包结构 |
| 1 | 分析原始文件（标注入口、敏感入参） |
| 2 | 拆分模块（copy-edit） |
| 3 | 创建 Golden 测试框架 |
| 4 | 验证与自动修复循环 |
| **5** | **用户确认与反馈沉淀** |

---

## 用户交互模式

### 步骤 1 后：分析结果确认

```
📋 原始文件分析完成：

入口识别：
- CPU 入口：`cpu_xxx_impl`
- NPU 入口：`npu_xxx_impl`（若不同名）

敏感入参（需 NPU wrapper 处理）：
- start_pos: list → Tensor
- cu_seqlens: list → Tensor

❓ 请确认：
1. 入口识别是否正确？
2. 是否有遗漏的敏感入参？

回复 "确认" 继续，或提供修正。
```

### 步骤 4 后：验证结果报告

```
✅ 验证完成 (尝试 N 次)

校验结果摘要：
- CPU Golden: PASS
- NPU Golden: PASS (shape check)
- API Golden: PASS

CSV 导出：golden_results_xxx.csv

❓ 以下点可能需要确认：
1. [若有 NPU shape 差异但 pass，说明原因]
2. [若有修复记录，列出]

回复 "继续" 进入规划阶段，或提供调整建议。
```

---

## 反馈沉淀

用户反馈将更新到对应文件：

| 反馈类型 | 更新位置 |
|---------|---------|
| 入口修正 | `test_cases.py` 的 entry 配置 |
| 敏感入参补充 | `npu.py` wrapper 逻辑 |
| Shape 检查调整 | `test_cases.py` 的 npu_check 配置 |
| 容差调整 | `test_cases.py` 的 tols 配置 |

---

## Golden 测试框架

### 核心策略

| 策略 | 说明 |
|------|------|
| CPU Golden | `original.cpu == refactor.cpu` |
| API Golden | 强制 `device="cpu"` 对比 |
| NPU Golden | 只做"可运行 + shape 检查" |
| CSV 导出 | 每个 case 的详细校验结果 |

### CSV 格式

| 列 | 说明 |
|---|------|
| case_name | 用例名 |
| cpu_status | PASS/FAIL/SKIP/TIMEOUT |
| npu_status | PASS/FAIL/SKIP/TIMEOUT |
| api_status | PASS/FAIL/SKIP/TIMEOUT |
| overall_status | PASS/FAIL |

---

## 验证与修复循环

```bash
# 验证命令
python -m op_<opname>.test
```

自动修复策略：
- P0 SyntaxError → 修复语法
- P1 ImportError → 改用包内绝对路径
- P2 NameError → 添加遗漏 import
- P3 AttributeError → 对齐入口函数名

最多 5 次尝试，超出则提示用户介入。

---

## 代码模板

见 `reference/` 目录：
- [case_schema.py](reference/case_schema.py)
- [compare.py](reference/compare.py)
- [adapters.py](reference/adapters.py)
- [golden_runner.py](reference/golden_runner.py)
- [test_cases.py](reference/test_cases.py)

---

## 最终报告

```
✅ 重构完成 [算子名]

模块化包：
- op_<opname>/utils.py: X 行
- op_<opname>/cpu.py: Y 行
- op_<opname>/npu.py: Z 行
- op_<opname>/api.py: M 行

验证结果：
- 尝试次数：N
- 最终状态：PASS
- CSV：golden_results_xxx.csv

用户反馈已沉淀：
- [列出用户的调整建议及更新位置]
```
