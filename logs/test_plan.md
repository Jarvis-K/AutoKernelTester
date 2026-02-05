# 测试计划：custom_layer_norm

## 算子信息

| 属性 | 值 |
|------|-----|
| 算子名称 | `custom_layer_norm` |
| 算子类型 | Layer Normalization (层归一化) |
| 测试文件 | `test_custom_layer_norm.py` |

---

## 1. 基础测试（必选）

**目标**：复现原 `__main__` 的测试结果，验证测试代码正确性

```yaml
baseline:
  name: "baseline"
  description: "复现原 __main__ 测试"
  config:
    batch_size: 4
    seq_len: 128
    hidden_size: 256
    normalized_shape: [256]
    weight: "ones(hidden_size)"
    bias: "zeros(hidden_size)"
    eps: 1.0e-5
    dtype: "float32"
  tolerance:
    atol: 1.0e-5
    rtol: null
  purpose: "验证测试框架能正确复现原测试结果"
```

---

## 2. 扩展测试

### 2.1 形状覆盖

测试不同的张量形状，覆盖小、中、大尺寸：

| 名称 | batch | seq | hidden | 说明 |
|------|-------|-----|--------|------|
| small_shape | 1 | 16 | 64 | 最小有效形状 |
| medium_shape | 2 | 64 | 128 | 中等形状 |
| large_shape | 8 | 512 | 1024 | 大规模形状 |

### 2.2 数据类型覆盖

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 1e-5 | - | 默认精度 |
| float16 | 1e-3 | - | 半精度（FP16） |
| bfloat16 | 2e-3 | - | 脑浮点（BF16） |

> 注：半精度容差需适当放宽

### 2.3 分支参数覆盖

测试可选参数的各种组合：

| 测试名 | weight | bias | 说明 |
|--------|--------|------|------|
| no_weight_bias | None | None | 不应用缩放和偏移 |
| with_weight_only | random | None | 仅应用缩放 |
| with_bias_only | None | random | 仅应用偏移 |
| with_weight_bias | random | random | 应用缩放和偏移 |

### 2.4 边界条件

| 测试名 | 配置 | 说明 |
|--------|------|------|
| single_element | B=1, S=1, H=1 | 单元素张量 |
| batch_one | B=1, S=128, H=256 | batch=1 |
| seq_one | B=4, S=1, H=256 | seq_len=1 |

### 2.5 特殊参数值

| 测试名 | eps | 说明 |
|--------|-----|------|
| eps_small | 1e-8 | 较小 eps 值 |
| eps_large | 1e-3 | 较大 eps 值 |

---

## 3. 测试用例汇总

| 类别 | 用例数 | 测试名 |
|------|--------|--------|
| 基础测试 | 1 | baseline |
| 形状覆盖 | 3 | small_shape, medium_shape, large_shape |
| 数据类型 | 2 | fp16, bf16 |
| 分支参数 | 4 | no_weight_bias, with_weight_only, with_bias_only, with_weight_bias |
| 边界条件 | 3 | single_element, batch_one, seq_one |
| 特殊参数 | 2 | eps_small, eps_large |
| **总计** | **15** | |

---

## 4. 测试函数签名

```python
def make_inputs(batch_size, seq_len, hidden_size, dtype=torch.float32, seed=42):
    """构造测试输入"""
    pass

def run_cpu(x, normalized_shape, weight, bias, eps):
    """运行 CPU 版本"""
    pass

def run_npu(x, normalized_shape, weight, bias, eps):
    """运行 NPU 版本"""
    pass

def compare(cpu_out, npu_out, atol):
    """对比结果，返回 max_diff 和是否通过"""
    pass
```

---

## 5. 用户调整记录

_（初始版本，暂无调整）_
