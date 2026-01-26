#!/usr/bin/env python3
"""
Example Operator File for Testing

This file demonstrates the expected format for user-provided operator files.
It contains a CPU reference implementation, NPU implementation, and simple test.
"""

import torch
# import torch_npu  # Uncomment when running on NPU


def custom_layer_norm_cpu(x: torch.Tensor, 
                          normalized_shape: tuple,
                          weight: torch.Tensor = None, 
                          bias: torch.Tensor = None,
                          eps: float = 1e-5) -> torch.Tensor:
    """
    CPU reference implementation of Layer Normalization.
    
    Args:
        x: Input tensor of shape (*, normalized_shape)
        normalized_shape: Shape over which to normalize
        weight: Optional learnable scale parameter
        bias: Optional learnable bias parameter
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of the same shape as input
    """
    # Calculate mean and variance over normalized dimensions
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    
    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # Apply scale and bias
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def custom_layer_norm_npu(x: torch.Tensor,
                          normalized_shape: tuple,
                          weight: torch.Tensor = None,
                          bias: torch.Tensor = None,
                          eps: float = 1e-5) -> torch.Tensor:
    """
    NPU accelerated implementation of Layer Normalization.
    
    In real usage, this would call an optimized NPU kernel.
    For demonstration, we use PyTorch's native implementation.
    """
    # In real NPU code: return torch_npu.npu_layer_norm(x, normalized_shape, weight, bias, eps)
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)


# Simple test code
if __name__ == "__main__":
    # Test configuration
    batch_size = 4
    seq_len = 128
    hidden_size = 256
    normalized_shape = (hidden_size,)
    
    # Create test tensors
    x = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.ones(hidden_size)
    bias = torch.zeros(hidden_size)
    
    # Run CPU reference
    cpu_out = custom_layer_norm_cpu(x, normalized_shape, weight, bias)
    
    # Run NPU implementation (on CPU for demo)
    npu_out = custom_layer_norm_npu(x, normalized_shape, weight, bias)
    
    # Compare
    max_diff = (cpu_out - npu_out).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    print(f"Test {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
