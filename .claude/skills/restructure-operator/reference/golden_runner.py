"""
Golden Runner - 测试运行器

关键特性：
- 每个 case 子进程执行 + timeout
- 入口映射显式来自 case（不再在 runner 里硬编码 if/else）
- NPU 默认只检查 shape/structure，不做 allclose
- API 强制 device="cpu" 做严格对比
- CSV 导出每个 case 的校验结果

使用时需将 op_<opname> 替换为实际包名
"""
import csv
import importlib
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from op_<opname>.testing.case_schema import GoldenCase
from op_<opname>.testing.compare import assert_same, assert_same_exception
from op_<opname>.testing.adapters import to_npu_inputs

@dataclass
class CaseResult:
    """单个 case 的测试结果"""
    name: str
    cpu_status: str = ""      # PASS / FAIL / SKIP / ERROR
    cpu_message: str = ""
    npu_status: str = ""
    npu_message: str = ""
    api_status: str = ""
    api_message: str = ""
    duration_s: float = 0.0

@dataclass
class RunSummary:
    total: int
    passed: int
    failed: int
    failed_cases: List[str]
    case_results: List[CaseResult] = field(default_factory=list)
    csv_path: str = ""

def _call(fn, args, kwargs):
    return fn(*args, **kwargs)

def _run_one_case(case: GoldenCase, q: mp.Queue):
    """运行单个 case，返回详细结果"""
    import time
    start_time = time.time()
    
    result = CaseResult(name=case.name)
    
    O = importlib.import_module("op_<opname>.original")
    CPU = importlib.import_module("op_<opname>.cpu")
    NPU = importlib.import_module("op_<opname>.npu")
    API = importlib.import_module("op_<opname>.api")

    args, kwargs = case.make_inputs()

    # baseline (original)
    baseline_ok = True
    baseline_out = None
    baseline_exc: Optional[BaseException] = None
    try:
        baseline_fn = getattr(O, case.orig_entry)
        baseline_out = _call(baseline_fn, args, kwargs)
    except BaseException as e:
        baseline_ok = False
        baseline_exc = e

    # expectation check
    if case.expect_exception_type is not None:
        if baseline_ok:
            q.put(("fail", f"expected exception {case.expect_exception_type}, but baseline returned", result))
            return
        if type(baseline_exc).__name__ != case.expect_exception_type:
            q.put(("fail", f"baseline exception type {type(baseline_exc).__name__} != expected {case.expect_exception_type}", result))
            return

    all_passed = True

    # CPU strict compare
    if case.run_cpu:
        try:
            out_cpu = _call(getattr(CPU, case.cpu_entry), args, kwargs)
            if baseline_ok:
                assert_same(out_cpu, baseline_out, case.tols.atol, case.tols.rtol, path="cpu")
                result.cpu_status = "PASS"
                result.cpu_message = "OK"
            else:
                result.cpu_status = "FAIL"
                result.cpu_message = "baseline raised, but cpu returned normally"
                all_passed = False
        except BaseException as e_cpu:
            if baseline_ok:
                result.cpu_status = "FAIL"
                result.cpu_message = str(e_cpu)
                all_passed = False
            else:
                try:
                    assert_same_exception(baseline_exc, e_cpu, strict_message=case.strict_exception_message)
                    result.cpu_status = "PASS"
                    result.cpu_message = f"Exception matched: {type(e_cpu).__name__}"
                except AssertionError as ae:
                    result.cpu_status = "FAIL"
                    result.cpu_message = str(ae)
                    all_passed = False
    else:
        result.cpu_status = "SKIP"
        result.cpu_message = "run_cpu=False"

    # NPU: run + check
    if case.run_npu:
        try:
            npu_args, npu_kwargs = to_npu_inputs(args, kwargs, device="npu")
            out_npu = _call(getattr(NPU, case.npu_entry), npu_args, npu_kwargs)
            if baseline_ok and case.npu_check != "none":
                def shape_of(x):
                    try:
                        import torch
                        if isinstance(x, torch.Tensor):
                            return (tuple(x.shape), str(x.dtype))
                    except Exception:
                        pass
                    return None

                if case.npu_check == "shape":
                    if shape_of(out_npu) != shape_of(baseline_out):
                        raise AssertionError(f"npu shape/dtype != baseline: {shape_of(out_npu)} vs {shape_of(baseline_out)}")
                elif case.npu_check == "structure":
                    if type(out_npu) is not type(baseline_out):
                        raise AssertionError(f"npu structure type mismatch: {type(out_npu)} vs {type(baseline_out)}")
            
            result.npu_status = "PASS"
            result.npu_message = f"OK (check={case.npu_check})"
        except BaseException as e_npu:
            if case.expect_exception_type is not None and type(e_npu).__name__ == case.expect_exception_type:
                result.npu_status = "PASS"
                result.npu_message = f"Exception matched: {type(e_npu).__name__}"
            elif not baseline_ok:
                result.npu_status = "PASS"
                result.npu_message = "baseline also raised"
            else:
                result.npu_status = "FAIL"
                result.npu_message = str(e_npu)
                all_passed = False
    else:
        result.npu_status = "SKIP"
        result.npu_message = "run_npu=False"

    # API strict compare on CPU
    if case.run_api:
        try:
            api_kwargs = dict(kwargs)
            api_kwargs["device"] = case.api_device_for_golden
            out_api = _call(getattr(API, case.api_entry), args, api_kwargs)
            if baseline_ok:
                assert_same(out_api, baseline_out, case.tols.atol, case.tols.rtol, path="api(cpu)")
                result.api_status = "PASS"
                result.api_message = "OK"
            else:
                result.api_status = "FAIL"
                result.api_message = "baseline raised, but api returned normally"
                all_passed = False
        except BaseException as e_api:
            if baseline_ok:
                result.api_status = "FAIL"
                result.api_message = str(e_api)
                all_passed = False
            else:
                try:
                    assert_same_exception(baseline_exc, e_api, strict_message=case.strict_exception_message)
                    result.api_status = "PASS"
                    result.api_message = f"Exception matched: {type(e_api).__name__}"
                except AssertionError as ae:
                    result.api_status = "FAIL"
                    result.api_message = str(ae)
                    all_passed = False
    else:
        result.api_status = "SKIP"
        result.api_message = "run_api=False"

    result.duration_s = time.time() - start_time
    
    if all_passed:
        q.put(("pass", "ok", result))
    else:
        q.put(("fail", "some checks failed", result))

def export_csv(case_results: List[CaseResult], csv_path: str):
    """导出测试结果到 CSV 文件"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'case_name', 
            'cpu_status', 'cpu_message',
            'npu_status', 'npu_message',
            'api_status', 'api_message',
            'duration_s', 'overall_status'
        ])
        writer.writeheader()
        for r in case_results:
            overall = "PASS" if all(s in ("PASS", "SKIP") for s in [r.cpu_status, r.npu_status, r.api_status]) else "FAIL"
            writer.writerow({
                'case_name': r.name,
                'cpu_status': r.cpu_status,
                'cpu_message': r.cpu_message[:200],  # 截断过长的消息
                'npu_status': r.npu_status,
                'npu_message': r.npu_message[:200],
                'api_status': r.api_status,
                'api_message': r.api_message[:200],
                'duration_s': f"{r.duration_s:.3f}",
                'overall_status': overall
            })
    print(f"\n📄 CSV exported: {csv_path}")

def run_cases(cases: List[GoldenCase], verbose: bool = True, csv_path: str = None) -> RunSummary:
    """
    运行所有 cases 并导出结果
    
    Args:
        cases: 测试用例列表
        verbose: 是否打印详细输出
        csv_path: CSV 导出路径，如果为 None 则自动生成
    """
    failed: List[str] = []
    passed = 0
    case_results: List[CaseResult] = []

    ctx = mp.get_context("spawn")

    for case in cases:
        if verbose:
            print(f"\n[CASE] {case.name}")

        q = ctx.Queue()
        p = ctx.Process(target=_run_one_case, args=(case, q))
        p.start()
        p.join(timeout=case.timeout_s)

        if p.is_alive():
            p.terminate()
            p.join()
            failed.append(case.name)
            result = CaseResult(
                name=case.name,
                cpu_status="TIMEOUT",
                cpu_message=f"timeout after {case.timeout_s}s",
                npu_status="TIMEOUT",
                npu_message=f"timeout after {case.timeout_s}s",
                api_status="TIMEOUT",
                api_message=f"timeout after {case.timeout_s}s",
                duration_s=case.timeout_s
            )
            case_results.append(result)
            if verbose:
                print(f"FAIL: timeout after {case.timeout_s}s")
            continue

        if not q.empty():
            status, msg, result = q.get()
        else:
            status, msg = "fail", "no result (crash?)"
            result = CaseResult(
                name=case.name,
                cpu_status="ERROR",
                cpu_message="process crashed",
                npu_status="ERROR",
                npu_message="process crashed",
                api_status="ERROR",
                api_message="process crashed"
            )
        
        case_results.append(result)
        
        if status == "pass":
            passed += 1
            if verbose:
                print(f"PASS (cpu={result.cpu_status}, npu={result.npu_status}, api={result.api_status})")
        else:
            failed.append(case.name)
            if verbose:
                print(f"FAIL: {msg}")
                if result.cpu_status == "FAIL":
                    print(f"  CPU: {result.cpu_message[:100]}")
                if result.npu_status == "FAIL":
                    print(f"  NPU: {result.npu_message[:100]}")
                if result.api_status == "FAIL":
                    print(f"  API: {result.api_message[:100]}")

    # 自动生成 CSV 路径
    if csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"golden_results_{timestamp}.csv"
    
    # 导出 CSV
    export_csv(case_results, csv_path)

    summary = RunSummary(
        total=len(cases), 
        passed=passed, 
        failed=len(failed), 
        failed_cases=failed,
        case_results=case_results,
        csv_path=csv_path
    )

    print("\n========== GOLDEN SUMMARY ==========")
    print(f"TOTAL: {summary.total}")
    print(f"PASS : {summary.passed}")
    print(f"FAIL : {summary.failed}")
    print(f"CSV  : {summary.csv_path}")
    if summary.failed_cases:
        print("FAILED CASES:")
        for n in summary.failed_cases:
            print(f" - {n}")
    if summary.failed == 0:
        print("✅ PASS: original == cpu == npu(run+shape) == api(cpu)")

    return summary
