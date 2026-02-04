"""
NPU 适配器 - 处理 list→tensor、device 放置

把常见的 NPU 入参问题框架化：
- list -> tensor（常见：start_pos/cu_seqlens/seqused 等）
- tensor -> npu device
- dtype 修正（如需要 int32）

注意：真正的 NPU wrapper 也要做；这里是双保险，避免某些入口绕过 wrapper
"""
from typing import Any, Dict, Tuple

def to_npu_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any], device: str = "npu"):
    try:
        import torch
    except Exception:
        return args, kwargs

    def conv(x):
        # list -> tensor（常见：start_pos/cu_seqlens/seqused 等）
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.int32).to(device)
        if isinstance(x, torch.Tensor) and x.device.type != device:
            return x.to(device)
        return x

    new_args = tuple(conv(a) for a in args)
    new_kwargs = {k: conv(v) for k, v in kwargs.items()}
    return new_args, new_kwargs
