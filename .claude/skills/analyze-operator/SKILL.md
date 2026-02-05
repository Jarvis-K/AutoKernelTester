---
name: analyze-operator
description: 分析目标算子文件，提取入口函数和测试逻辑
---

# 分析算子

**目的**：阅读目标算子文件，提取关键信息，为后续测试做准备。

---

## 输出产物

- `logs/analyze_result.md` - 分析结果文档

---

## 执行步骤

| 步骤 | 说明 |
|------|------|
| 1 | 阅读原算子文件完整内容 |
| 2 | 识别 CPU/NPU 入口函数 |
| 3 | 提取 `__main__` 测试逻辑 |
| 4 | 记录输入构造方式 |
| 5 | **生成 `logs/analyze_result.md`** |
| 6 | 展示分析结果，请求用户确认 |

---

## 需提取的信息

### 1. 入口函数

```python
# 识别模式
def xxx_cpu(...):    # CPU 实现
def xxx_npu(...):    # NPU 实现
def xxx(...):        # 通用入口（可能内部判断设备）
```

### 2. `__main__` 测试逻辑

```python
if __name__ == "__main__":
    # 需要提取的内容：
    # - 输入张量构造方式
    # - 函数调用方式
    # - 结果验证逻辑（如何对比 CPU/NPU）
    # - 使用的配置参数（shape、dtype、容差等）
```

### 3. 输入参数映射

| 参数名 | 类型 | 默认值 | 含义 |
|--------|------|--------|------|
| `batch_size` | int | ? | 批大小 |
| `seq_len` | int | ? | 序列长度 |
| ... | ... | ... | ... |

---

## 输出格式

分析完成后，向用户展示：

```
📖 算子分析结果

文件：<算子文件路径>

入口函数：
- CPU: xxx_cpu(x, weight, bias, ...)
- NPU: xxx_npu(x, weight, bias, ...)

__main__ 测试逻辑：
- 输入构造：torch.randn(batch, seq, hidden)
- 调用方式：直接调用 xxx_cpu/xxx_npu
- 验证方法：torch.allclose + max_diff

默认配置：
- batch_size=4, seq_len=128, hidden=256
- dtype=float32
- atol=1e-5, rtol=1e-5

❓ 请确认：
1. 入口函数识别是否正确？
2. 是否有遗漏的参数？

回复 "确认" 继续规划阶段。
```

---

## 注意事项

> [!IMPORTANT]
> 1. **只读不改**：此阶段不修改任何文件
> 2. **完整提取**：确保 `__main__` 中所有关键信息都被记录
> 3. **识别不确定点**：如有模糊之处，主动询问用户
