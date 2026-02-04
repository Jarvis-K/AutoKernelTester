"""
Golden Test Case Schema - 用例模型定义

关键更新点：
- entry 显式拆分：orig_entry / cpu_entry / npu_entry / api_entry
- NPU 检查策略：默认 npu_check="shape"（可选 "none"|"shape"|"structure"）
- 设备策略：api_device_for_golden="cpu" 固化避免精度差异
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Literal

@dataclass(frozen=True)
class Tolerances:
    atol: float = 1e-5
    rtol: float = 1e-5

NPUCheck = Literal["none", "shape", "structure"]

@dataclass(frozen=True)
class GoldenCase:
    name: str

    # 允许不同模块不同入口名（解决 cpu_compressor vs npu_compressor）
    orig_entry: str
    cpu_entry: str
    npu_entry: str
    api_entry: str = "op"

    make_inputs: Callable[[], Tuple[Tuple[Any, ...], Dict[str, Any]]]
    tols: Tolerances = Tolerances()

    run_cpu: bool = True
    run_npu: bool = True
    run_api: bool = True

    # API golden 强制走 CPU，避免默认走 NPU 的精度差
    api_device_for_golden: str = "cpu"

    # NPU 默认不做数值对比（解决 CPU vs NPU precision mismatch）
    npu_check: NPUCheck = "shape"

    # 异常一致性
    expect_exception_type: Optional[str] = None
    strict_exception_message: bool = True

    # 每个 case 的超时（秒）
    timeout_s: int = 60
