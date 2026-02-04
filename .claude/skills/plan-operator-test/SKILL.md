---
name: plan-operator-test
description: 基于参数说明自动推断 prefill/decode 测试计划
---

# 操作符测试规划

基于用户提供的 **参数说明（param_spec）** 自动推断测试计划。

---

## 输入要求

用户需提供参数说明，包含：
- 参数名
- 参数类型（int/float/bool/enum/tensor/optional）
- 参数含义简述

---

## 自动推断逻辑

### 执行模式推断

| 信号词 | 推断模式 |
|-------|---------|
| `start_pos` / `cache` / `kv` / `state` | decode |
| `seq_len` / `cu_seqlens` / `prefill_length` | prefill |
| 无状态/无序列 | 简单场景 |

### 形状轴推断

自动识别：batch_size、seq_len、head_dim、hidden_size 等

默认值（若未指定）：
- B: [1, 2]
- S: [1, 128]
- decode_step: [1, 8]

### 分支参数推断

- bool → 覆盖 [True, False]
- enum → 覆盖 2-3 个典型值
- 连续参数 → 覆盖 [default, small, large]

---

## 用户交互模式

### 推断完成后的确认

```
📋 测试计划已生成：logs/test_op_plan.md

推断结果：
- 执行模式：prefill + decode
- 形状覆盖：B=[1,2], S=[1,128], step=[1,8]
- 数据类型：fp16, bf16, fp32
- 分支参数：use_cache=[True,False], causal=[True,False]

❓ 以下推断可能需要确认：

1. 执行模式推断
   - 检测到 `start_pos` 参数 → 推断支持 decode
   - 这是否正确？

2. 形状范围
   - seq_len 最大设为 128
   - 是否需要覆盖更长序列？

3. 容差设置
   - fp16: atol=1e-2, rtol=1e-2
   - 是否需要调整？

请回复：
- "批准" - 开始执行测试
- "调整 seq_len 到 512" - 我会更新计划
- "增加 xxx 场景" - 我会补充
```

---

## 测试场景生成

### Prefill 场景

- P0_minimal: B=1, S=1 (sanity check)
- P1_typical: B=1, S=128 (典型场景)
- P2_boundary: 边界值测试

### Decode 场景

- D0_single_step: B=1, step=1
- D1_multi_step: step=8 (状态增长)
- D2_multi_batch: B=2, step=8

### 异常场景（若有限制）

- EX_invalid_shape: 测试形状限制

---

## 容差策略

| dtype | atol | rtol |
|-------|------|------|
| fp32 | 1e-6 | 1e-6 |
| fp16 | 1e-2 | 1e-2 |
| bf16 | 2e-2 | 2e-2 |

若涉及 softmax/norm → 自动放宽。

---

## 输出产物

- `logs/test_op_plan.md` - 中文测试计划
- 结构化 CaseSpec（供 execute 阶段使用）

---

## 反馈沉淀

用户反馈将更新到计划中：

| 反馈类型 | 更新位置 |
|---------|---------|
| 执行模式修正 | 覆盖轴表 |
| 形状调整 | shapes 配置 |
| 容差调整 | tolerance 配置 |
| 场景补充 | CaseSpec 列表 |
| 分支参数 | branch_params 表 |

---

## 后续步骤

```
计划已保存。

下一步：使用 `/execute-operator-test` 执行测试。

或者回复调整建议，我会更新计划后重新展示。
```