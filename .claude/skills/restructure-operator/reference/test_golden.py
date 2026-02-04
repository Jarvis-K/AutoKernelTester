"""
pytest 集成 - Golden 测试入口

使用时需将 op_<opname> 替换为实际包名
"""
from op_<opname>.test_cases import build_cases
from op_<opname>.testing.golden_runner import run_cases

def test_golden():
    summary = run_cases(build_cases(), verbose=True)
    assert summary.failed == 0, f"failed cases: {summary.failed_cases}"
