# 算子分析结果

## 基本信息

| 属性 | 值 |
|------|-----|
| 文件路径 | `sample_operator.py` |
| 算子名称 | `custom_layer_norm` |
| 算子类型 | Layer Normalization (层归一化) |

---

## 入口函数

### CPU 实现入口

```python
def custom_layer_norm_cpu(x: torch.Tensor,
                          normalized_shape: tuple,
                          weight: torch.Tensor = None,
                          bias: torch.Tensor = None,
                          eps: float = 1e-5) -> torch.Tensor
```

**位置**: `sample_operator.py:13-45`

### NPU 实现入口

```python
def custom_layer_norm_npu(x: torch.Tensor,
                          normalized_shape: tuple,
                          weight: torch.Tensor = None,
                          bias: torch.Tensor = None,
                          eps: float = 1e-5) -> torch.Tensor
```

**位置**: `sample_operator.py:48-60`

---

## `__main__` 测试逻辑提取

### 测试配置

| 参数 | 值 |
|------|-----|
| `batch_size` | 4 |
| `seq_len` | 128 |
| `hidden_size` | 256 |
| `normalized_shape` | `(256,)` |
| `eps` | 1e-5 (默认值) |

### 输入构造方式

```python
# 输入张量
x = torch.randn(batch_size, seq_len, hidden_size)
# shape: (4, 128, 256)

# 可选参数
weight = torch.ones(hidden_size)      # 全1缩放参数
bias = torch.zeros(hidden_size)       # 全0偏移参数
```

### 函数调用方式

```python
# CPU 调用
cpu_out = custom_layer_norm_cpu(x, normalized_shape, weight, bias)

# NPU 调用
npu_out = custom_layer_norm_npu(x, normalized_shape, weight, bias)
```

### 结果验证逻辑

```python
max_diff = (cpu_out - npu_out).abs().max().item()
print(f"Max difference: {max_diff:.2e}")
print(f"Test {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
```

**验证标准**:
- 计算方式: 逐元素绝对差值的最大值
- 通过阈值: `max_diff < 1e-5`
- 输出格式: 科学计数法 (`.2e`)

---

## 参数映射表

| 参数名 | 类型 | 默认值 | 含义 | 示例值 |
|--------|------|--------|------|--------|
| `x` | torch.Tensor | 必填 | 输入张量 | `torch.randn(4, 128, 256)` |
| `normalized_shape` | tuple | 必填 | 归一化维度 | `(256,)` |
| `weight` | torch.Tensor | `None` | 可学习的缩放参数 | `torch.ones(256)` |
| `bias` | torch.Tensor | `None` | 可学习的偏移参数 | `torch.zeros(256)` |
| `eps` | float | `1e-5` | 数值稳定性常数 | `1e-5` |

---

## 关键发现

1. **函数签名完全一致**: CPU 和 NPU 版本的参数列表相同
2. **normalized_shape 是关键**: 决定在哪些维度上进行归一化
3. **weight/bias 可选**: 原测试使用全1和全0，相当于不应用缩放和偏移
4. **验证容差**: `1e-5` 是绝对误差容限（无 rtol）

---

## 待确认事项

- [ ] 是否需要测试 `weight=None` 和 `bias=None` 的情况？
- [ ] 是否需要测试不同的 `eps` 值？
- [ ] 是否需要测试不同的 `normalized_shape`（如多维归一化）？
