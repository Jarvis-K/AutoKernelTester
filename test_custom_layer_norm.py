#!/usr/bin/env python3
"""
test_custom_layer_norm.py - Custom Layer Norm 算子测试

输入抽象化设计：
- make_inputs(**config) 统一构造输入
- run_cpu/run_npu 封装调用
- 不同测试仅修改 TEST_CONFIGS 参数
"""

import torch
import sys
from sample_operator import custom_layer_norm_cpu, custom_layer_norm_npu

# ============ 配置 ============
ATOL, RTOL = 1e-5, None

# ============ 输入抽象层 ============
def make_inputs(
    batch_size=4,
    seq_len=128,
    hidden_size=256,
    dtype=torch.float32,
    weight_mode="ones",
    bias_mode="zeros",
    eps=1e-5,
    seed=42,
    **kwargs
):
    """
    统一输入构造接口

    设计原则：
    - 所有参数都有默认值（来自原 __main__）
    - 不同测试仅传递需要变化的参数
    - 返回 (args, kwargs) 元组供调用使用

    Args:
        batch_size: 批大小
        seq_len: 序列长度
        hidden_size: 隐藏维度
        dtype: 数据类型
        weight_mode: "ones", "random", 或 None
        bias_mode: "zeros", "random", 或 None
        eps: 数值稳定性常数
        seed: 随机种子
    """
    torch.manual_seed(seed)

    # 构造输入（复制自原 __main__）
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    # 构造 weight
    if weight_mode == "ones":
        weight = torch.ones(hidden_size, dtype=dtype)
    elif weight_mode == "random":
        weight = torch.randn(hidden_size, dtype=dtype)
    else:  # None
        weight = None

    # 构造 bias
    if bias_mode == "zeros":
        bias = torch.zeros(hidden_size, dtype=dtype)
    elif bias_mode == "random":
        bias = torch.randn(hidden_size, dtype=dtype)
    else:  # None
        bias = None

    normalized_shape = (hidden_size,)

    return {
        "x": x,
        "normalized_shape": normalized_shape,
        "weight": weight,
        "bias": bias,
        "eps": eps
    }


# ============ 封装调用 ============
def run_cpu(**config):
    """运行 CPU 实现"""
    inputs = make_inputs(**config)
    return custom_layer_norm_cpu(
        inputs["x"],
        inputs["normalized_shape"],
        inputs["weight"],
        inputs["bias"],
        inputs["eps"]
    )


def run_npu(**config):
    """运行 NPU 实现"""
    try:
        import torch_npu
    except ImportError:
        # NPU 不可用时，回退到 CPU（用于演示）
        inputs = make_inputs(**config)
        return custom_layer_norm_npu(
            inputs["x"],
            inputs["normalized_shape"],
            inputs["weight"],
            inputs["bias"],
            inputs["eps"]
        )

    inputs = make_inputs(**config)
    x_npu = inputs["x"].npu()
    weight_npu = inputs["weight"].npu() if inputs["weight"] is not None else None
    bias_npu = inputs["bias"].npu() if inputs["bias"] is not None else None

    return custom_layer_norm_npu(
        x_npu,
        inputs["normalized_shape"],
        weight_npu,
        bias_npu,
        inputs["eps"]
    )


def compare(cpu_out, npu_out, atol=ATOL, rtol=RTOL):
    """
    对比结果

    Returns:
        (status, max_diff): 状态字符串和最大差异
    """
    if npu_out is None:
        return "SKIP", None

    npu_cpu = npu_out.cpu() if npu_out.device.type != "cpu" else npu_out
    max_diff = (cpu_out - npu_cpu).abs().max().item()

    if rtol is not None:
        passed = torch.allclose(cpu_out, npu_cpu, atol=atol, rtol=rtol)
    else:
        passed = max_diff < atol

    return ("PASS" if passed else "FAIL"), max_diff


# ============ 测试配置 ============
# 初始只包含 baseline（复现 __main__）
TEST_CONFIGS = [
    {
        "name": "baseline",
        "batch_size": 4,
        "seq_len": 128,
        "hidden_size": 256,
        "weight_mode": "ones",
        "bias_mode": "zeros",
        "eps": 1e-5,
        "dtype": torch.float32,
        "atol": 1e-5,
    },
]


# ============ 运行测试 ============
def run_all():
    """运行所有测试配置"""
    results = []
    print("=" * 60)
    print("Running custom_layer_norm tests")
    print("=" * 60)

    for cfg in TEST_CONFIGS:
        name = cfg.pop("name")
        atol = cfg.pop("atol", ATOL)
        rtol = cfg.pop("rtol", RTOL)

        print(f"\n[{name}]")
        print(f"  Config: batch={cfg.get('batch_size')}, seq={cfg.get('seq_len')}, hidden={cfg.get('hidden_size')}")

        cpu_out = run_cpu(**cfg)
        npu_out = run_npu(**cfg)

        status, diff = compare(cpu_out, npu_out, atol=atol, rtol=rtol)

        print(f"  Status: {status}")
        if diff is not None:
            print(f"  Max diff: {diff:.2e}")
            print(f"  Tolerance: atol={atol}" + (f", rtol={rtol}" if rtol else ""))

        results.append({
            "name": name,
            "status": status,
            "max_diff": diff,
            "config": cfg.copy(),
            "atol": atol,
            "rtol": rtol
        })

        cfg["name"] = name
        cfg["atol"] = atol
        cfg["rtol"] = rtol

    return results


def main():
    results = run_all()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: max_diff={r['max_diff']:.2e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
