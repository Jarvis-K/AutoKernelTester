---
name: write-and-verify
description: 编写测试文件，迭代验证直至复现原测试
---

# 编写并验证测试

**目的**：生成 `test_<opname>.py`，复现原 `__main__` 测试，迭代直至结果一致。

---

## 前置条件

- 已完成 `/analyze-operator`（入口函数已识别）
- 已完成 `/plan-and-confirm`（测试计划已确认）

---

## 核心设计：输入抽象化

> [!IMPORTANT]
> **所有输入通过 `make_inputs()` 函数构造**
> 
> 不同测试用例仅改变配置参数，不改变代码结构

---

## 测试文件模板

```python
#!/usr/bin/env python3
"""
test_<opname>.py - 算子测试文件

输入抽象化设计：
- make_inputs(**config) 统一构造输入
- run_cpu/run_npu 封装调用
- 不同测试仅修改 TEST_CONFIGS 参数
"""
import torch
from <original> import <cpu_func>, <npu_func>

# ============ 配置 ============
ATOL, RTOL = 1e-5, 1e-5

# ============ 输入抽象层 ============
def make_inputs(
    batch_size=4,     # 从 __main__ 提取的默认值
    seq_len=128,
    hidden_size=256,
    dtype=torch.float32,
    seed=0,
    **kwargs         # 允许扩展参数
):
    """
    统一输入构造接口
    
    设计原则：
    - 所有参数都有默认值（来自原 __main__）
    - 不同测试仅传递需要变化的参数
    - 返回 (args, kwargs) 元组供调用使用
    """
    torch.manual_seed(seed)
    
    # 构造输入（复制自原 __main__）
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    weight = torch.ones(hidden_size, dtype=dtype)
    bias = torch.zeros(hidden_size, dtype=dtype)
    
    return (x, weight, bias), {}

# ============ 封装调用 ============
def run_cpu(**config):
    """运行 CPU 实现"""
    args, kwargs = make_inputs(**config)
    return <cpu_func>(*args, **kwargs)

def run_npu(**config):
    """运行 NPU 实现"""
    try:
        import torch_npu
    except ImportError:
        return None
    
    args, kwargs = make_inputs(**config)
    npu_args = tuple(a.npu() if isinstance(a, torch.Tensor) else a for a in args)
    return <npu_func>(*npu_args, **kwargs)

def compare(cpu_out, npu_out):
    """对比结果"""
    if npu_out is None:
        return "SKIP", None
    npu_cpu = npu_out.cpu()
    max_diff = (cpu_out - npu_cpu).abs().max().item()
    if torch.allclose(cpu_out, npu_cpu, atol=ATOL, rtol=RTOL):
        return "PASS", max_diff
    return "FAIL", max_diff

# ============ 测试配置 ============
# 初始只包含 baseline（复现 __main__）
TEST_CONFIGS = [
    {"name": "baseline", "batch_size": 4, "seq_len": 128, "hidden_size": 256},
]

# ============ 运行测试 ============
def run_all():
    results = []
    for cfg in TEST_CONFIGS:
        name = cfg.pop("name")
        cpu_out = run_cpu(**cfg)
        npu_out = run_npu(**cfg)
        status, diff = compare(cpu_out, npu_out)
        print(f"[{name}] {status}" + (f" (diff={diff:.2e})" if diff else ""))
        results.append({"name": name, "status": status, "diff": diff, **cfg})
        cfg["name"] = name
    return results

if __name__ == "__main__":
    run_all()
```

---

## 验证流程

### 步骤 1：生成测试文件

根据分析结果，生成 `test_<opname>.py`

### 步骤 2：运行 baseline 测试

```bash
python test_<opname>.py
```

### 步骤 3：对比验证

| 检查项 | 期望 |
|--------|------|
| 脚本能否运行 | 无报错 |
| baseline 结果 | PASS |
| 输出形状 | 与原 `__main__` 一致 |
| 数值差异 | 在容差范围内 |

### 步骤 4：迭代修正

**如果不通过**：

```
❌ Baseline 验证失败

问题：max_diff=0.05，超出容差 atol=1e-5

原 __main__ 输出：
- shape: (4, 128, 256)
- 前5个值: [0.123, -0.456, ...]

当前测试输出：
- shape: (4, 128, 256)
- 前5个值: [0.128, -0.451, ...]

可能原因：
1. 输入构造不一致
2. 随机种子不同
3. 函数参数遗漏

正在尝试修正...
```

**修正循环**：
1. 分析差异原因
2. 修改 `make_inputs()` 或调用方式
3. 重新运行验证
4. 重复直至通过

> [!CAUTION]
> **最多尝试 5 次**。如仍不通过，向用户报告问题并请求帮助。

---

## 验证通过后

```
✅ Baseline 验证通过

测试文件：test_<opname>.py
结果：PASS (max_diff=1.2e-7)

已验证：
- 输入构造与原 __main__ 一致
- CPU/NPU 调用正确
- 结果在容差范围内

❓ 请确认后继续扩展测试阶段。
```

---

## 用户交互

每次验证结果都需要展示给用户：
- 通过 → 询问是否继续
- 失败 → 展示差异，询问如何处理
