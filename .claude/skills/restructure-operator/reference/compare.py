"""
统一对比模块 - 严格结构/数值/异常对比

覆盖：
- 结构一致（list/tuple/dict 递归）
- torch tensor：dtype/shape/allclose
- 异常一致（类型 + message 可选严格）
"""
from typing import Any
import math

def assert_same_exception(e1: BaseException, e2: BaseException, strict_message: bool = True):
    if type(e1) is not type(e2):
        raise AssertionError(f"Exception type mismatch: {type(e1)} vs {type(e2)}")
    if strict_message and str(e1) != str(e2):
        raise AssertionError(f"Exception message mismatch:\n- {e1}\n- {e2}")

def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)

def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))

def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False

def assert_same(a: Any, b: Any, atol: float, rtol: float, path: str = "out"):
    if _is_mapping(a) or _is_mapping(b):
        if not (_is_mapping(a) and _is_mapping(b)):
            raise AssertionError(f"{path}: type mismatch {type(a)} vs {type(b)}")
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"{path}: keys mismatch {set(a.keys())} vs {set(b.keys())}")
        for k in a:
            assert_same(a[k], b[k], atol, rtol, path=f"{path}[{k!r}]")
        return

    if _is_sequence(a) or _is_sequence(b):
        if not (_is_sequence(a) and _is_sequence(b)):
            raise AssertionError(f"{path}: type mismatch {type(a)} vs {type(b)}")
        if len(a) != len(b):
            raise AssertionError(f"{path}: len mismatch {len(a)} vs {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            assert_same(ai, bi, atol, rtol, path=f"{path}[{i}]")
        return

    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"{path}: None mismatch {a} vs {b}")
        return

    if _is_torch_tensor(a) or _is_torch_tensor(b):
        import torch
        if not (_is_torch_tensor(a) and _is_torch_tensor(b)):
            raise AssertionError(f"{path}: tensor type mismatch {type(a)} vs {type(b)}")
        if a.dtype != b.dtype:
            raise AssertionError(f"{path}: dtype mismatch {a.dtype} vs {b.dtype}")
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"{path}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.numel() == 0:
            return
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{path}: value mismatch max_diff={diff} atol={atol} rtol={rtol}")
        return

    # 数值标量：允许容差
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isfinite(a) and math.isfinite(b):
            if abs(a - b) > (atol + rtol * abs(b)):
                raise AssertionError(f"{path}: scalar mismatch {a} vs {b} atol={atol} rtol={rtol}")
            return

    if a != b:
        raise AssertionError(f"{path}: value mismatch {a} vs {b}")
