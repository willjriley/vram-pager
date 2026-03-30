"""M6: Test VRAM pager with a real model inference.

Loads a small model, replaces Linear layers with PagedLinear,
runs inference, and compares output to standard inference.
"""
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
import torch as _t; _tlib = os.path.join(os.path.dirname(_t.__file__), "lib")
if os.path.exists(_tlib): os.add_dll_directory(_tlib)

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctypes
import torch
import torch.nn as nn
import time
from pager import VRAMPager, replace_linear_with_paged

# Load our compiled CUDA kernel
dll = ctypes.CDLL(r"C:\repos\vram_pager\build\dequant.dll")
dll.launch_dequant_int8.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]


class CUDAKernelWrapper:
    """Wrapper that matches the interface VRAMPager expects for its kernel."""
    def __init__(self, dll):
        self.dll = dll

    def dequantize_int8(self, quantized, scales, block_size):
        n = quantized.numel()
        output = torch.empty(n, dtype=torch.float32, device=quantized.device)
        self.dll.launch_dequant_int8(
            quantized.data_ptr(), scales.data_ptr(), output.data_ptr(),
            n, block_size, None
        )
        return output


kernel = CUDAKernelWrapper(dll)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================
# Test 1: Simple MLP — verify paged inference matches standard
# ============================================================
print("=" * 60)
print("TEST 1: Simple MLP — Correctness Verification")
print("=" * 60)

class SimpleMLP(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleMLP(1024).cuda().eval()
x = torch.randn(1, 64, 1024, device="cuda")

# Standard inference
with torch.no_grad():
    standard_output = model(x).clone()

# Convert to paged
pager = VRAMPager(block_size=128)
pager.set_kernel(kernel)

# Save original weights before replacement
original_weights = {name: param.data.clone() for name, param in model.named_parameters()}

model_paged = replace_linear_with_paged(model, pager)

# Paged inference
with torch.no_grad():
    paged_output = model_paged(x)

diff = (standard_output - paged_output).abs()
print(f"Max diff: {diff.max().item():.6f}")
print(f"Mean diff: {diff.mean().item():.6f}")
print(f"Relative error: {(diff / standard_output.abs().clamp(min=1e-8)).mean().item():.6f}")
print(f"Output matches: {diff.max().item() < 0.1}")  # INT8 has ~0.02 max error per layer
print()

# ============================================================
# Test 2: Simulated Transformer Block — Speed Test
# ============================================================
print("=" * 60)
print("TEST 2: Transformer Block — Speed Comparison")
print("=" * 60)

class TransformerBlock(nn.Module):
    def __init__(self, dim=5120, ffn_dim=13824):
        super().__init__()
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        qkv = self.attn_qkv(x)
        h = self.attn_out(qkv[..., :x.shape[-1]])
        x = x + h
        x = x + self.ffn_down(torch.relu(self.ffn_up(x)))
        return x

# Standard: weight on CPU, offload per forward
block_std = TransformerBlock().eval()
block_std.cpu()  # simulate offloaded state
x = torch.randn(1, 32, 5120, device="cuda")

# Standard offload timing
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    block_std.cuda()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = block_std(x)
    torch.cuda.synchronize()
    block_std.cpu()
    torch.cuda.synchronize()
standard_ms = (time.time() - t0) / 10 * 1000

# Paged version
block_paged = TransformerBlock().cuda().eval()
pager2 = VRAMPager(block_size=128)
pager2.set_kernel(kernel)
block_paged = replace_linear_with_paged(block_paged, pager2)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = block_paged(x)
    torch.cuda.synchronize()
paged_ms = (time.time() - t0) / 10 * 1000

print(f"Standard (CPU offload): {standard_ms:.1f} ms/block")
print(f"Paged (INT8 + kernel):  {paged_ms:.1f} ms/block")
print(f"Speedup:                {standard_ms/paged_ms:.2f}x")
print()
print(f"Projected for 40 blocks (Wan 14B):")
print(f"  Standard: {standard_ms * 40 / 1000:.1f}s/step")
print(f"  Paged:    {paged_ms * 40 / 1000:.1f}s/step")
print(f"  Saved:    {(standard_ms - paged_ms) * 40 / 1000:.1f}s/step")
print(f"  20 steps: {(standard_ms - paged_ms) * 40 * 20 / 1000:.0f}s total saved")

# ============================================================
# Test 3: Memory Usage
# ============================================================
print()
print("=" * 60)
print("TEST 3: VRAM Usage Comparison")
print("=" * 60)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Standard: load full block to GPU
block_mem = TransformerBlock().cuda()
torch.cuda.synchronize()
standard_vram = torch.cuda.max_memory_allocated() / 1e6
del block_mem
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Paged: weights in CPU RAM, only decompressed on demand
block_paged2 = TransformerBlock().cuda().eval()
pager3 = VRAMPager(block_size=128)
pager3.set_kernel(kernel)
block_paged2 = replace_linear_with_paged(block_paged2, pager3)

with torch.no_grad():
    _ = block_paged2(x)
torch.cuda.synchronize()
paged_vram = torch.cuda.max_memory_allocated() / 1e6

print(f"Standard VRAM (full block on GPU): {standard_vram:.0f} MB")
print(f"Paged VRAM (on-demand decompress): {paged_vram:.0f} MB")
print(f"VRAM reduction: {(1 - paged_vram/standard_vram)*100:.0f}%")
print()
print(f"Pager storage: {pager3.summary()}")
