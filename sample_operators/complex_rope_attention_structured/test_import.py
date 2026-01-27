"""
Quick verification test for restructured complex_rope_attention module.
"""
import sys
sys.path.insert(0, 'sample_operators')

import torch

# Import from restructured module
from complex_rope_attention_structured import (
    rope_attention_cpu,
    rope_attention_npu,
    _compute_rope_frequencies,
    MAX_SEQ_LEN,
)

# Test parameters
batch_size = 2
num_heads = 4
seq_len = 64
head_dim = 64

print("Testing restructured complex_rope_attention module...")

# Create test inputs
query = torch.randn(batch_size, num_heads, seq_len, head_dim)
key = torch.randn(batch_size, num_heads, seq_len, head_dim)
value = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compute RoPE frequencies
freqs = _compute_rope_frequencies(head_dim, MAX_SEQ_LEN)
cos_cached = torch.cos(freqs)
sin_cached = torch.sin(freqs)

print(f"  Input shapes: query={query.shape}, key={key.shape}, value={value.shape}")
print(f"  RoPE cache shapes: cos={cos_cached.shape}, sin={sin_cached.shape}")

# Test CPU function
try:
    output_cpu, weights_cpu = rope_attention_cpu(
        query, key, value, cos_cached, sin_cached, is_causal=True
    )
    print(f"  ✓ CPU function works: output={output_cpu.shape}, weights={weights_cpu.shape}")
except Exception as e:
    print(f"  ✗ CPU function failed: {e}")

# Test NPU function (will use fallback if no NPU)
try:
    output_npu, weights_npu = rope_attention_npu(
        query, key, value, cos_cached, sin_cached, is_causal=True
    )
    print(f"  ✓ NPU function works: output={output_npu.shape}, weights={weights_npu.shape}")
except Exception as e:
    print(f"  ✗ NPU function failed: {e}")

print("\n✓ Restructured module verification complete!")
