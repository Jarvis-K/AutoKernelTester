---
name: restructure-operator
description: 从原算子文件提取测试代码，封装为可复用的测试模块
---

# 操作符重构

**目的**：从原文件的 `__main__` 测试代码**快速复现**测试，**封装原函数调用**，**隐藏不必要的细节**。

---

## 核心原则

> [!IMPORTANT]
> **复用而非重写**：
> - ✅ 直接调用原文件中的函数
> - ✅ 提取 `__main__` 中的测试逻辑
> - ✅ 封装为简洁的测试接口
> - ❌ 不拷贝实现代码
> - ❌ 不修改原文件

---

## 输出结构

```
<same_dir>/
├── <original>.py        # 保持不动
└── test_<opname>.py     # 封装后的测试模块
```

---

## 执行流程

| 步骤 | 说明 |
|------|------|
| 1 | **分析原文件**：识别入口函数 + 提取 `__main__` 测试逻辑 |
| 2 | **生成测试模块**：封装调用，隐藏细节 |
| 3 | **验证运行**：`python test_<opname>.py` |
| 4 | **用户确认** |

---

## 步骤 1：分析原文件

从原文件提取：

1. **入口函数**：`xxx_cpu()` / `xxx_npu()` 签名
2. **测试逻辑**：`__main__` 中的输入构造、调用、验证代码
3. **配置参数**：形状、dtype、容差等

### 示例原文件

```python
# original.py
def custom_layer_norm_cpu(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    ...

def custom_layer_norm_npu(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    ...

if __name__ == "__main__":
    # 测试配置
    batch_size, seq_len, hidden_size = 4, 128, 256
    normalized_shape = (hidden_size,)
    
    # 输入构造
    x = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.ones(hidden_size)
    bias = torch.zeros(hidden_size)
    
    # 调用测试
    cpu_out = custom_layer_norm_cpu(x, normalized_shape, weight, bias)
    npu_out = custom_layer_norm_npu(x, normalized_shape, weight, bias)
    
    # 验证
    max_diff = (cpu_out - npu_out).abs().max().item()
    print(f"Test {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
```

---

## 步骤 2：生成测试模块

**关键**：封装 `__main__` 逻辑为可复用函数，隐藏输入构造细节。

```python
#!/usr/bin/env python3
"""
test_<opname>.py - 封装原文件测试逻辑

复用原文件函数，提供简洁测试接口：
- run_cpu(config) -> result
- run_npu(config) -> result  
- run_all() -> 完整测试
"""
import torch
from <original> import <cpu_func>, <npu_func>

# ============ 配置 ============
ATOL, RTOL = 1e-5, 1e-5

# ============ 输入构造（封装自 __main__）============
def make_inputs(batch_size=4, seq_len=128, hidden_size=256, dtype=torch.float32, seed=0):
    """
    封装原 __main__ 的输入构造逻辑
    
    从原文件提取的默认参数：
    - batch_size=4, seq_len=128, hidden_size=256 (来自 __main__)
    """
    torch.manual_seed(seed)
    normalized_shape = (hidden_size,)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    weight = torch.ones(hidden_size, dtype=dtype)
    bias = torch.zeros(hidden_size, dtype=dtype)
    return (x, normalized_shape, weight, bias), {}

# ============ 封装调用接口 ============
def run_cpu(**config):
    """运行 CPU 实现"""
    args, kwargs = make_inputs(**config)
    return <cpu_func>(*args, **kwargs)

def run_npu(**config):
    """运行 NPU 实现（自动处理设备转移）"""
    try:
        import torch_npu
    except ImportError:
        return None
    
    args, kwargs = make_inputs(**config)
    npu_args = tuple(a.npu() if isinstance(a, torch.Tensor) else a for a in args)
    return <npu_func>(*npu_args, **kwargs)

def compare(cpu_out, npu_out):
    """对比 CPU/NPU 结果"""
    if npu_out is None:
        return "SKIP"
    npu_cpu = npu_out.cpu()
    if torch.allclose(cpu_out, npu_cpu, atol=ATOL, rtol=RTOL):
        return "PASS"
    return f"FAIL (max_diff={(cpu_out - npu_cpu).abs().max().item():.2e})"

# ============ 测试用例 ============
TEST_CONFIGS = [
    {"name": "default", "batch_size": 4, "seq_len": 128, "hidden_size": 256},
    {"name": "small", "batch_size": 1, "seq_len": 16, "hidden_size": 64},
    {"name": "large", "batch_size": 8, "seq_len": 512, "hidden_size": 512},
]

def run_all():
    """运行所有测试"""
    print("=" * 50)
    print("Golden Test: <opname>")
    print("=" * 50)
    
    for cfg in TEST_CONFIGS:
        name = cfg.pop("name")
        cpu_out = run_cpu(**cfg)
        npu_out = run_npu(**cfg)
        status = compare(cpu_out, npu_out)
        print(f"[{name}] CPU shape: {cpu_out.shape}, NPU: {status}")
        cfg["name"] = name  # restore

if __name__ == "__main__":
    run_all()
```

---

## 封装要点

| 原 `__main__` 代码 | 封装后 |
|-------------------|--------|
| 输入构造逻辑 | `make_inputs(**config)` |
| CPU 调用 | `run_cpu(**config)` |
| NPU 调用 | `run_npu(**config)` (含设备转移) |
| 结果验证 | `compare(cpu, npu)` |
| 配置参数 | `TEST_CONFIGS` 列表 |

---

## 步骤 3：验证运行

```bash
python test_<opname>.py
```

预期输出：
```
==================================================
Golden Test: <opname>
==================================================
[default] CPU shape: torch.Size([4, 128, 256]), NPU: PASS
[small] CPU shape: torch.Size([1, 16, 64]), NPU: PASS
[large] CPU shape: torch.Size([8, 512, 512]), NPU: PASS
```

---

## 步骤 4：用户确认

```
✅ 测试模块生成完成

生成文件：test_<opname>.py

封装内容：
- make_inputs(): 从 __main__ 提取的输入构造
- run_cpu()/run_npu(): 封装的调用接口
- TEST_CONFIGS: 测试配置列表

❓ 请确认：
1. 输入构造是否正确反映原 __main__ 逻辑？
2. 是否需要调整测试配置？

回复 "确认" 继续。
```

---

## 反馈沉淀

| 反馈 | 更新位置 |
|-----|---------|
| 输入构造调整 | `make_inputs()` 函数 |
| 配置修改 | `TEST_CONFIGS` 列表 |
| 容差调整 | `ATOL`/`RTOL` 常量 |
