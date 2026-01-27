#!/usr/bin/env python3
"""
Precision Test: complex_rope_attention (Quick Smoke Suite)

Tests: ~50 configurations
Estimated Time: ~30 seconds
Focus: Basic functionality verification
"""

import sys
sys.path.insert(0, 'sample_operators')

import torch
import csv
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import from restructured module
from complex_rope_attention_structured import (
    rope_attention_cpu,
    rope_attention_npu,
    _compute_rope_frequencies,
    MAX_SEQ_LEN,
)

# =============================================================================
# TEST CONFIGURATIONS (Quick Smoke Suite)
# =============================================================================

# Quick Smoke: Batch sizes [1, 2], Sequence lengths [64, 128], Head dims [64]
# Dtypes: [float32] only, No masks, causal=False, Pattern: random only

QUICK_SMOKE_CONFIGS = [
    # Batch size variations (12 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 1, "heads": 16, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 16, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 1, "heads": 8, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 8, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 1, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},

    # Causal mode tests (6 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": True, "pattern": "random", "masks": "causal_only"},
    {"batch": 2, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": True, "pattern": "random", "masks": "causal_only"},
    {"batch": 1, "heads": 8, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": True, "pattern": "random", "masks": "causal_only"},
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": True, "pattern": "random", "masks": "causal_only"},

    # Cross-attention tests (6 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 8, "seq_q": 64, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 1, "heads": 16, "seq_q": 128, "seq_k": 256, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 256, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "none"},

    # Edge cases - zeros (4 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "zeros", "masks": "none"},
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "zeros", "masks": "none"},

    # Edge cases - ones (4 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "ones", "masks": "none"},
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "ones", "masks": "none"},

    # Edge cases - very_small (4 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "very_small", "masks": "none"},
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "very_small", "masks": "none"},

    # Edge cases - very_large (4 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "very_large", "masks": "none"},
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "very_large", "masks": "none"},

    # Edge cases - mixed_sign (4 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "mixed_sign", "masks": "none"},
    {"batch": 1, "heads": 1, "seq_q": 1, "seq_k": 1, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "mixed_sign", "masks": "none"},

    # User mask tests (6 tests)
    {"batch": 1, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "user_only"},
    {"batch": 2, "heads": 8, "seq_q": 64, "seq_k": 64, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "user_only"},
    {"batch": 1, "heads": 8, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "user_only"},
    {"batch": 2, "heads": 16, "seq_q": 128, "seq_k": 128, "head_dim": 64, "dtype": torch.float32, "causal": False, "pattern": "random", "masks": "user_only"},
]

print(f"Total test configurations: {len(QUICK_SMOKE_CONFIGS)}")

# =============================================================================
# TOLERANCE SETTINGS
# =============================================================================

TOLERANCES = {
    torch.float32: {"rtol": 1e-3, "atol": 1e-4},
    torch.float16: {"rtol": 1e-2, "atol": 1e-3},
    torch.bfloat16: {"rtol": 5e-2, "atol": 1e-2},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_tensor(shape: tuple, dtype: torch.dtype, pattern: str) -> torch.Tensor:
    """Generate test tensor with specified pattern."""
    if pattern == "random":
        return torch.randn(shape, dtype=dtype)
    elif pattern == "zeros":
        return torch.zeros(shape, dtype=dtype)
    elif pattern == "ones":
        return torch.ones(shape, dtype=dtype)
    elif pattern == "very_small":
        return torch.randn(shape, dtype=dtype) * 1e-7
    elif pattern == "very_large":
        return torch.randn(shape, dtype=dtype) * 1e5
    elif pattern == "mixed_sign":
        # Mix of positive and negative values
        return torch.randn(shape, dtype=dtype) * torch.randint(-1, 2, shape).to(dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def create_user_mask(seq_len_q: int, seq_len_k: int, batch: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a simple user attention mask."""
    # Create a 2D mask that blocks some positions
    mask = torch.zeros(seq_len_q, seq_len_k, dtype=dtype)
    # Block upper right quadrant
    mid_q = seq_len_q // 2
    mid_k = seq_len_k // 2
    mask[mid_q:, mid_k:] = float('-inf')
    # Return 2D mask - the merge function will expand it correctly
    return mask


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> Dict[str, float]:
    """Compute precision metrics between reference and test outputs."""
    ref = ref.float().cpu()
    test = test.float().cpu()

    # Check for NaN/Inf
    if torch.isnan(ref).any() or torch.isinf(ref).any():
        return {"has_nan_inf": True, "max_abs_diff": float('inf'), "max_rel_diff": float('inf')}
    if torch.isnan(test).any() or torch.isinf(test).any():
        return {"has_nan_inf": True, "max_abs_diff": float('inf'), "max_rel_diff": float('inf')}

    diff = torch.abs(ref - test)
    denom = torch.abs(ref).clamp(min=1e-8)

    return {
        "has_nan_inf": False,
        "max_abs_diff": diff.max().item(),
        "max_rel_diff": (diff / denom).max().item(),
        "mean_abs_diff": diff.mean().item(),
        "mse": torch.mean((ref - test) ** 2).item(),
    }


def run_single_test(config: Dict, test_id: int) -> Dict:
    """Run a single test configuration."""
    result = {
        "test_id": test_id,
        "config": config.copy(),
        "status": "PASS",
        "error": None,
        "metrics": {},
    }

    try:
        # Extract config parameters
        batch = config["batch"]
        heads = config["heads"]
        seq_q = config["seq_q"]
        seq_k = config["seq_k"]
        head_dim = config["head_dim"]
        dtype = config["dtype"]
        is_causal = config["causal"]
        pattern = config.get("pattern", "random")
        mask_type = config.get("masks", "none")

        # Generate input tensors
        query = generate_tensor((batch, heads, seq_q, head_dim), dtype, pattern)
        key = generate_tensor((batch, heads, seq_k, head_dim), dtype, pattern)
        value = generate_tensor((batch, heads, seq_k, head_dim), dtype, pattern)

        # Compute RoPE frequencies
        max_seq = max(seq_q, seq_k)
        freqs = _compute_rope_frequencies(head_dim, max_seq)
        cos_cached = torch.cos(freqs)
        sin_cached = torch.sin(freqs)

        # Setup masks
        attention_mask = None
        key_padding_mask = None

        if mask_type == "user_only" or mask_type == "combined":
            attention_mask = create_user_mask(seq_q, seq_k, batch, dtype)

        # Run CPU reference
        cpu_output, cpu_weights = rope_attention_cpu(
            query, key, value, cos_cached, sin_cached,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            training=False,
            use_flash=False,
        )

        # Run NPU implementation (will use naive path on CPU)
        npu_output, npu_weights = rope_attention_npu(
            query, key, value, cos_cached, sin_cached,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            training=False,
            use_flash=False,  # Force naive path for comparison
        )

        # Verify shapes
        if cpu_output.shape != npu_output.shape:
            result["status"] = "FAIL"
            result["error"] = f"Shape mismatch: CPU {cpu_output.shape} vs NPU {npu_output.shape}"
            return result

        # Compute metrics for output
        output_metrics = compute_metrics(cpu_output, npu_output)
        result["metrics"]["output"] = output_metrics

        # Compute metrics for attention weights
        weights_metrics = compute_metrics(cpu_weights, npu_weights)
        result["metrics"]["weights"] = weights_metrics

        # Check tolerances
        tol = TOLERANCES[dtype]
        output_passed = (
            not output_metrics["has_nan_inf"] and
            output_metrics["max_abs_diff"] <= tol["atol"] or
            output_metrics["max_rel_diff"] <= tol["rtol"]
        )
        weights_passed = (
            not weights_metrics["has_nan_inf"] and
            weights_metrics["max_abs_diff"] <= tol["atol"] or
            weights_metrics["max_rel_diff"] <= tol["rtol"]
        )

        if output_passed and weights_passed:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error"] = f"Tolerance exceeded: output_max_abs={output_metrics['max_abs_diff']:.2e}, weights_max_abs={weights_metrics['max_abs_diff']:.2e}"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def run_tests() -> Dict:
    """Run all tests and return results."""
    results = {
        "total": len(QUICK_SMOKE_CONFIGS),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": [],
    }

    print(f"\n{'='*60}")
    print(f"Running Quick Smoke Test Suite")
    print(f"{'='*60}")
    print(f"Total tests: {results['total']}")
    print(f"Est. time: ~30 seconds")
    print(f"{'='*60}\n")

    for i, config in enumerate(QUICK_SMOKE_CONFIGS, 1):
        result = run_single_test(config, i)
        results["details"].append(result)

        # Update counters
        if result["status"] == "PASS":
            results["passed"] += 1
            print(f"[{i:2d}/{results['total']}] PASS - batch={config['batch']}, heads={config['heads']}, seq=({config['seq_q']},{config['seq_k']}), pattern={config.get('pattern', 'random')}, masks={config.get('masks', 'none')}")
        elif result["status"] == "FAIL":
            results["failed"] += 1
            print(f"[{i:2d}/{results['total']}] FAIL - batch={config['batch']}, heads={config['heads']}, seq=({config['seq_q']},{config['seq_k']}), pattern={config.get('pattern', 'random')}, masks={config.get('masks', 'none')}")
            print(f"       Error: {result['error']}")
        else:
            results["errors"] += 1
            print(f"[{i:2d}/{results['total']}] ERROR - batch={config['batch']}, heads={config['heads']}, seq=({config['seq_q']},{config['seq_k']})")
            print(f"       Error: {result['error']}")

    return results


def export_csv(results: Dict, path: str):
    """Export results to CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'test_id', 'batch', 'heads', 'seq_q', 'seq_k', 'head_dim',
            'dtype', 'causal', 'pattern', 'masks', 'status',
            'output_max_abs_diff', 'output_max_rel_diff', 'output_mse',
            'weights_max_abs_diff', 'weights_max_rel_diff', 'weights_mse',
            'error'
        ])
        writer.writeheader()

        for d in results['details']:
            config = d['config']
            writer.writerow({
                'test_id': d['test_id'],
                'batch': config['batch'],
                'heads': config['heads'],
                'seq_q': config['seq_q'],
                'seq_k': config['seq_k'],
                'head_dim': config['head_dim'],
                'dtype': str(config['dtype']),
                'causal': config['causal'],
                'pattern': config.get('pattern', 'random'),
                'masks': config.get('masks', 'none'),
                'status': d['status'],
                'output_max_abs_diff': d.get('metrics', {}).get('output', {}).get('max_abs_diff', ''),
                'output_max_rel_diff': d.get('metrics', {}).get('output', {}).get('max_rel_diff', ''),
                'output_mse': d.get('metrics', {}).get('output', {}).get('mse', ''),
                'weights_max_abs_diff': d.get('metrics', {}).get('weights', {}).get('max_abs_diff', ''),
                'weights_max_rel_diff': d.get('metrics', {}).get('weights', {}).get('max_rel_diff', ''),
                'weights_mse': d.get('metrics', {}).get('weights', {}).get('mse', ''),
                'error': d.get('error', '')
            })

    print(f"\nCSV exported to: {path}")


def generate_report(results: Dict, path: str):
    """Generate markdown report."""
    total = results['total']
    passed = results['passed']
    failed = results['failed']
    errors = results['errors']
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Group by pattern
    pattern_stats = {}
    for d in results['details']:
        pattern = d['config'].get('pattern', 'random')
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {'total': 0, 'passed': 0, 'failed': 0}
        pattern_stats[pattern]['total'] += 1
        if d['status'] == 'PASS':
            pattern_stats[pattern]['passed'] += 1
        elif d['status'] == 'FAIL':
            pattern_stats[pattern]['failed'] += 1

    # Group by mask type
    mask_stats = {}
    for d in results['details']:
        mask_type = d['config'].get('masks', 'none')
        if mask_type not in mask_stats:
            mask_stats[mask_type] = {'total': 0, 'passed': 0, 'failed': 0}
        mask_stats[mask_type]['total'] += 1
        if d['status'] == 'PASS':
            mask_stats[mask_type]['passed'] += 1
        elif d['status'] == 'FAIL':
            mask_stats[mask_type]['failed'] += 1

    report = f"""# Precision Test Report: complex_rope_attention

## Test Suite: Quick Smoke

**Date:** 2026-01-28
**Total Tests:** {total}
**Estimated Time:** ~30 seconds

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | {total} |
| **Passed** | {passed} |
| **Failed** | {failed} |
| **Errors** | {errors} |
| **Pass Rate** | {pass_rate:.1f}% |
| **CSV Export** | `complex_rope_attention_quick_results.csv` |

---

## Results by Pattern

| Pattern | Total | Passed | Failed | Pass Rate |
|---------|-------|--------|--------|-----------|
"""

    for pattern, stats in sorted(pattern_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report += f"| {pattern} | {stats['total']} | {stats['passed']} | {stats['failed']} | {rate:.1f}% |\n"

    report += "\n## Results by Mask Type\n\n"
    report += "| Mask Type | Total | Passed | Failed | Pass Rate |\n"
    report += "|-----------|-------|--------|--------|----------|\n"

    for mask_type, stats in sorted(mask_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report += f"| {mask_type} | {stats['total']} | {stats['passed']} | {stats['failed']} | {rate:.1f}% |\n"

    # Failed tests section
    if failed > 0 or errors > 0:
        report += "\n## Failed/Errored Tests\n\n"
        report += "| Test ID | Config | Status | Error |\n"
        report += "|---------|--------|--------|-------|\n"

        for d in results['details']:
            if d['status'] in ['FAIL', 'ERROR']:
                config = d['config']
                config_str = f"b={config['batch']}, h={config['heads']}, seq=({config['seq_q']},{config['seq_k']}), pattern={config.get('pattern', 'random')}, masks={config.get('masks', 'none')}"
                report += f"| {d['test_id']} | {config_str} | {d['status']} | {d.get('error', '')} |\n"

    report += f"""

## Test Configuration

**Quick Smoke Suite Parameters:**
- Batch sizes: `[1, 2]`
- Sequence lengths: `[64, 128]`
- Head dimensions: `[64]`
- Number of heads: `[8, 16]`
- Data types: `[float32]`
- Causal modes: `[True, False]`
- Mask combinations: `[none, causal_only, user_only]`
- Value patterns: `[random, zeros, ones, very_small, very_large, mixed_sign]`

## Tolerance Settings

| Dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-3 | 1e-4 |

## Recommendations

"""

    if pass_rate >= 95:
        report += "✅ **Pass rate ≥ 95%** - Operator is performing within expected tolerances.\n\n"
    elif pass_rate >= 80:
        report += "⚠️ **Pass rate 80-95%** - Some configurations failing. Review failed tests above.\n\n"
    else:
        report += "❌ **Pass rate < 80%** - Significant issues detected. Investigation required.\n\n"

    report += """
## Notes

- This is the **Quick Smoke** test suite for basic functionality verification
- Tests compare CPU reference vs NPU implementation (both using naive path on CPU)
- For comprehensive testing, use the full test suite with more configurations
- Flash attention path requires float16/bfloat16 and seq_len ≥ 1024

---

**Report generated:** 2026-01-28
"""

    with open(path, 'w') as f:
        f.write(report)

    print(f"Report exported to: {path}")


if __name__ == "__main__":
    results = run_tests()

    print(f"\n{'='*60}")
    print(f"Test Complete")
    print(f"{'='*60}")
    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Failed: {results['failed']}/{results['total']}")
    print(f"Errors: {results['errors']}/{results['total']}")
    print(f"Pass Rate: {(results['passed']/results['total']*100):.1f}%")
    print(f"{'='*60}\n")

    # Export results
    base_path = Path("sample_operators/complex_rope_attention_structured")
    export_csv(results, base_path / "complex_rope_attention_quick_results.csv")
    generate_report(results, base_path / "complex_rope_attention_quick_report.md")
