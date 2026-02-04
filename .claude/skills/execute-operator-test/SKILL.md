---
name: execute-operator-test
description: 生成测试用例、执行 Golden 测试、输出中文报告
---

# 执行操作符测试

从测试计划生成用例，运行 Golden 测试，输出详细报告。

---

## 前置条件

- 已完成 `/restructure-operator`（模块化包 + Golden 框架）
- 已完成 `/plan-operator-test`（测试计划 `logs/test_op_plan.md`）

---

## 执行步骤

### 步骤一：生成 test_cases.py

根据 `logs/test_op_plan.md` 中的 CaseSpec，更新 `op_<opname>/test_cases.py`

### 步骤二：执行测试

```bash
python -m op_<opname>.test
```

- 输出保存到：`logs/test_op_run.log`
- CSV 结果：`golden_results_<timestamp>.csv`

### 步骤三：生成报告

输出 `logs/test_op_report.md`（中文）

---

## 用户交互模式

### 执行完成后的报告

```
✅ 测试完成

📊 结果摘要：
- 总用例：N
- 通过：X（xx%）
- 失败：Y

覆盖情况：
- 模式：prefill ✓, decode ✓
- dtype：fp16 ✓, bf16 ✓, fp32 ✓
- 分支参数：use_cache ✓, causal ✓

❓ 失败用例分析（若有）：

1. D1_multi_step
   - 失败类型：numeric mismatch
   - max_diff：0.05（超出 atol=0.01）
   - 建议：可能需要放宽容差或检查算法差异

2. P2_boundary
   - 失败类型：shape mismatch
   - 期望：(2, 128, 64)，实际：(2, 64, 128)
   - 建议：检查 transpose 逻辑

请回复：
- "完成" - 结束测试流程
- "放宽 D1 容差到 0.1" - 我会调整并重跑
- "查看 P2 详情" - 我会展示详细对比
- "重跑失败用例" - 只重跑失败的
```

---

## 失败分类

| 类型 | 说明 | 建议 |
|-----|------|-----|
| import_fail | 导入失败 | 检查绝对导入 |
| baseline_error | 原始算子报错 | 检查输入有效性 |
| numeric_mismatch | 数值差异 | 放宽容差或检查算法 |
| shape_mismatch | 形状不匹配 | 检查 reshape/transpose |
| dtype_mismatch | 类型不匹配 | 检查类型转换 |
| exception_mismatch | 异常不一致 | 检查异常处理 |

---

## 输出产物

| 文件 | 说明 |
|-----|------|
| `op_<opname>/test_cases.py` | 测试用例（已更新） |
| `logs/test_op_run.log` | 运行日志 |
| `logs/test_op_report.md` | 中文报告 |
| `golden_results_<ts>.csv` | 详细结果 |

---

## 报告结构

```markdown
# 测试报告：[算子名]

## 运行信息
- 算子：xxx
- 计划：logs/test_op_plan.md
- 命令：python -m op_xxx.test
- 时间：2024-xx-xx
- 结果：N/M 通过

## 覆盖概览
- 模式：prefill ✓, decode ✓
- dtype：[列表]
- 分支参数：[列表]

## 失败摘要
[若有失败，详细列出]

## 复现方式
python -m op_xxx.test

## 结论
PASS / FAIL（需修复后复测）
```

---

## 反馈沉淀

| 反馈类型 | 更新位置 |
|---------|---------|
| 容差调整 | `test_cases.py` tols |
| 重跑请求 | 执行特定 case |
| 问题分析 | `logs/test_op_report.md` |

---

## 停止条件

完成后停止，不继续修改算子实现。

若需修复，提示用户：
```
测试发现差异，建议检查 npu.py 中的 xxx 逻辑。

修复后，使用 `/execute-operator-test` 重新测试。
```