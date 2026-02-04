"""
测试用例数据生成模板

要求：
- 只写 build_cases()
- 先写一个通用 make_inputs(...)，case 变体都调用它
- case 必须显式填好 orig/cpu/npu entry
- 对于 NPU 精度敏感的算子：保持 npu_check="shape"（默认）

使用时需将 op_<opname> 替换为实际包名
"""
from op_<opname>.testing.case_schema import GoldenCase, Tolerances

def build_cases():
    cases = []

    def make_inputs(B=1, S=16, seed=0):
        import torch
        torch.manual_seed(seed)
        x = torch.randn(B, S, dtype=torch.float32)
        start_pos = [S - 1] * B  # 故意用 list，验证 wrapper/adapter 能兜住
        kwargs = dict(start_pos=start_pos)
        return (x,), kwargs

    cases.append(GoldenCase(
        name="basic",
        orig_entry="cpu_compressor",   # original 入口
        cpu_entry="cpu_compressor",    # cpu.py 导出的入口（可同名）
        npu_entry="npu_compressor",    # npu.py 导出的入口（可不同名）
        make_inputs=lambda: make_inputs(B=2, S=32, seed=0),
        tols=Tolerances(atol=1e-5, rtol=1e-5),
        run_cpu=True,
        run_npu=True,
        run_api=True,
        npu_check="shape",
        timeout_s=60,
    ))

    return cases
