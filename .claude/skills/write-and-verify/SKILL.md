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

## 输出产物

- `test_<opname>.py` - 测试文件
- `test_config.json` - 测试配置文件
- `logs/verify_result.md` - 验证结果文档

---

## 核心设计：输入抽象化 + 配置分离

> [!IMPORTANT]
> 1. **所有输入通过 `make_inputs()` 函数构造**
> 2. **配置写入单独的 `test_config.json` 文件**
> 
> 不同测试用例仅改变 JSON 配置，不改变代码

---

## 配置文件格式

`test_config.json`:
```json
{
  "operator": "<opname>",
  "timeout": 60,
  "tolerance": {
    "atol": 1e-5,
    "rtol": 1e-5
  },
  "test_cases": [
    {
      "name": "baseline",
      "batch_size": 4,
      "seq_len": 128,
      "hidden_size": 256,
      "dtype": "float32"
    },
    {
      "name": "small",
      "batch_size": 1,
      "seq_len": 16,
      "hidden_size": 64
    },
    {
      "name": "large_with_custom_timeout",
      "batch_size": 8,
      "seq_len": 512,
      "hidden_size": 512,
      "timeout": 120
    }
  ]
}
```

---

## 测试文件模板

```python
#!/usr/bin/env python3
"""
test_<opname>.py - 算子测试文件

输入抽象化设计：
- make_inputs(**config) 统一构造输入
- 测试配置从 test_config.json 读取
"""
import json
import torch
from <original> import <cpu_func>, <npu_func>

# ============ 加载配置 ============
def load_config(config_file="test_config.json"):
    with open(config_file, "r") as f:
        return json.load(f)

CONFIG = load_config()
ATOL = CONFIG["tolerance"]["atol"]
RTOL = CONFIG["tolerance"]["rtol"]

# ============ dtype 映射 ============
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# ============ 输入抽象层 ============
def make_inputs(
    batch_size=4,
    seq_len=128,
    hidden_size=256,
    dtype="float32",
    seed=0,
    **kwargs
):
    torch.manual_seed(seed)
    dt = DTYPE_MAP.get(dtype, torch.float32)
    
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dt)
    weight = torch.ones(hidden_size, dtype=dt)
    bias = torch.zeros(hidden_size, dtype=dt)
    
    return (x, weight, bias), {}

# ============ 封装调用 ============
def run_cpu(**config):
    args, kwargs = make_inputs(**config)
    return <cpu_func>(*args, **kwargs)

def run_npu(**config):
    try:
        import torch_npu
    except ImportError:
        return None
    
    args, kwargs = make_inputs(**config)
    npu_args = tuple(a.npu() if isinstance(a, torch.Tensor) else a for a in args)
    return <npu_func>(*npu_args, **kwargs)

def compare(cpu_out, npu_out):
    if npu_out is None:
        return "SKIP", None
    npu_cpu = npu_out.cpu()
    max_diff = (cpu_out - npu_cpu).abs().max().item()
    if torch.allclose(cpu_out, npu_cpu, atol=ATOL, rtol=RTOL):
        return "PASS", max_diff
    return "FAIL", max_diff

# ============ 超时保护 ============
from multiprocessing import Process, Queue
import time

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    使用多进程运行函数，超时则终止
    
    Args:
        func: 要执行的函数
        timeout: 超时时间（秒）
        *args, **kwargs: 函数参数
    
    Returns:
        (success, result) - success=False 表示超时
    """
    def worker(q, func, args, kwargs):
        try:
            result = func(*args, **kwargs)
            q.put(("success", result))
        except Exception as e:
            q.put(("error", str(e)))
    
    q = Queue()
    p = Process(target=worker, args=(q, func, args, kwargs))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, None  # 超时
    
    if not q.empty():
        status, result = q.get()
        if status == "success":
            return True, result
        else:
            return False, result  # 异常
    
    return False, None

TIMEOUT = CONFIG.get("timeout", 60)  # 默认 60 秒超时

# ============ 运行测试 ============
def run_single_case(cfg):
    """运行单个测试用例"""
    cfg = cfg.copy()
    name = cfg.pop("name")
    timeout = cfg.pop("timeout", TIMEOUT)
    
    # CPU 测试
    success, cpu_out = run_with_timeout(run_cpu, timeout, **cfg)
    if not success:
        return {"name": name, "status": "TIMEOUT", "diff": None, **cfg}
    
    # NPU 测试
    success, npu_out = run_with_timeout(run_npu, timeout, **cfg)
    if not success:
        return {"name": name, "status": "TIMEOUT", "diff": None, **cfg}
    
    status, diff = compare(cpu_out, npu_out)
    return {"name": name, "status": status, "diff": diff, **cfg}

def run_all():
    results = []
    for cfg in CONFIG["test_cases"]:
        result = run_single_case(cfg)
        status = result["status"]
        diff = result.get("diff")
        print(f"[{result['name']}] {status}" + (f" (diff={diff:.2e})" if diff else ""))
        results.append(result)
    return results

if __name__ == "__main__":
    run_all()
```

---

## 验证流程（TDD）

> [!IMPORTANT]
> **TDD 验证：对比期望输出**
> 
> 加载 `logs/expected_output.json`，逐项对比实际输出

### 步骤 1：加载期望输出

```python
def load_expected():
    with open("logs/expected_output.json") as f:
        return json.load(f)

EXPECTED = load_expected()
```

### 步骤 2：运行 baseline 测试

```bash
python test_<opname>.py
```

### 步骤 3：逐项对比验证

```python
def verify_against_expected(cpu_out, npu_out, expected):
    """TDD 验证：逐项对比期望输出"""
    checks = []
    
    # 1. Shape 验证
    cpu_shape = list(cpu_out.shape)
    exp_shape = expected["expected_results"]["cpu_shape"]
    checks.append(("cpu_shape", cpu_shape == exp_shape, cpu_shape, exp_shape))
    
    # 2. Dtype 验证
    cpu_dtype = str(cpu_out.dtype)
    exp_dtype = expected["expected_results"]["cpu_dtype"]
    checks.append(("cpu_dtype", cpu_dtype == exp_dtype, cpu_dtype, exp_dtype))
    
    # 3. Sample values 验证（前5个值）
    cpu_first_5 = cpu_out.flatten()[:5].tolist()
    exp_first_5 = expected["sample_values"]["cpu_first_5"]
    values_close = all(abs(a - b) < 1e-5 for a, b in zip(cpu_first_5, exp_first_5))
    checks.append(("sample_values", values_close, cpu_first_5, exp_first_5))
    
    # 4. Max diff 验证
    if npu_out is not None:
        max_diff = (cpu_out - npu_out.cpu()).abs().max().item()
        exp_diff = expected["expected_results"]["max_diff"]
        diff_ok = max_diff <= exp_diff * 1.5  # 允许 50% 浮动
        checks.append(("max_diff", diff_ok, max_diff, exp_diff))
    
    return checks
```

### 步骤 4：输出验证报告

```
✅ TDD 验证结果

| 检查项 | 期望 | 实际 | 结果 |
|--------|------|------|------|
| cpu_shape | [4,128,256] | [4,128,256] | ✅ |
| cpu_dtype | torch.float32 | torch.float32 | ✅ |
| sample_values | [0.123,-0.456,...] | [0.123,-0.456,...] | ✅ |
| max_diff | 1.23e-7 | 1.21e-7 | ✅ |
```

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

> [!IMPORTANT]
> **修正原则**
> 
> 1. **不修改目标算子代码** - 只修改测试文件 `test_<opname>.py`
> 2. **从原实现对比推理** - 对比 `__main__` 和测试代码的差异：
>    - 输入张量的构造方式是否一致（shape、dtype、device）
>    - 随机种子是否相同
>    - 函数调用参数是否完整
>    - 额外的预处理/后处理是否遗漏
> 3. **逐步定位** - 先打印中间结果对比，缩小差异范围

**对比推理示例**：
```python
# 原 __main__ 的调用
out = custom_layer_norm_cpu(x, normalized_shape, weight, bias, eps=1e-5)

# 测试代码的调用 - 检查是否遗漏 eps 参数
out = cpu_func(*args, **kwargs)  # eps 是否传入？
```

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
