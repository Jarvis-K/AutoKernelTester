"""
CLI 入口 - 支持 python -m op_<opname>.test

使用时需将 op_<opname> 替换为实际包名
"""
import sys
import pytest

def main():
    # 只跑本包 tests，保持稳定
    ret = pytest.main(["-q", "op_<opname>/tests/test_golden.py"])
    sys.exit(ret)

if __name__ == "__main__":
    main()
