"""Test VRAM Pager with SDXL, Flux, and quality comparison."""
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

# Load CUDA kernel
dll = ctypes.CDLL(r"C:\repos\vram_pager\build\dequant.dll")
dll.launch_dequant_int8.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]

class CUDAKernelWrapper:
    def __init__(self, dll):
        self.dll = dll
    def dequantize_int8(self, quantized, scales, block_size):
        n = quantized.numel()
        output = torch.empty(n, dtype=torch.float32, device=quantized.device)
        self.dll.launch_dequant_int8(
            quantized.data_ptr(), scales.data_ptr(), output.data_ptr(),
            n, block_size, None)
        return output

kernel = CUDAKernelWrapper(dll)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA kernel loaded\n")

# ============================================================
# TEST 1: SDXL-scale model (2.6B params, dim=1280)
# ============================================================
print("=" * 60)
print("TEST 1: SDXL-Scale Model (dim=1280, 24 blocks)")
print("=" * 60)

class SDXLBlock(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, dim * 4, bias=False)
        self.ffn_down = nn.Linear(dim * 4, dim, bias=False)
    def forward(self, x):
        qkv = self.attn_qkv(x)
        h = self.attn_out(qkv[..., :x.shape[-1]])
        x = x + h
        x = x + self.ffn_down(torch.relu(self.ffn_up(x)))
        return x

sdxl_blocks = nn.ModuleList([SDXLBlock(1280) for _ in range(24)]).eval()
x = torch.randn(1, 64, 1280, device="cuda")

# Standard forward (all on GPU)
sdxl_blocks.cuda()
with torch.no_grad():
    standard_out = x.clone()
    for b in sdxl_blocks:
        standard_out = b(standard_out)
    standard_result = standard_out.clone()

# Paged forward
pager = VRAMPager(block_size=128)
pager.set_kernel(kernel)
for i, b in enumerate(sdxl_blocks):
    for name, param in b.named_parameters():
        pager.store_weight(f"blocks.{i}.{name}", param.data)

sdxl_blocks_paged = nn.ModuleList([SDXLBlock(1280) for _ in range(24)]).cuda().eval()
sdxl_blocks_paged = replace_linear_with_paged(sdxl_blocks_paged, pager)

with torch.no_grad():
    paged_out = x.clone()
    for b in sdxl_blocks_paged:
        paged_out = b(paged_out)
    paged_result = paged_out.clone()

diff = (standard_result - paged_result).abs()
print(f"Max error after 24 blocks: {diff.max().item():.6f}")
print(f"Mean error: {diff.mean().item():.6f}")
print(f"Relative error: {(diff / standard_result.abs().clamp(min=1e-8)).mean().item():.6f}")
print(f"SDXL-scale: {'PASS' if diff.max().item() < 1.0 else 'FAIL'}")
print(f"Pager storage: {pager.summary()}")

del sdxl_blocks, sdxl_blocks_paged, pager
torch.cuda.empty_cache()

# ============================================================
# TEST 2: Flux-scale model (dim=3072, 19 blocks)
# ============================================================
print()
print("=" * 60)
print("TEST 2: Flux-Scale Model (dim=3072, 19 blocks)")
print("=" * 60)

flux_blocks = nn.ModuleList([SDXLBlock(3072) for _ in range(19)]).eval()
x2 = torch.randn(1, 32, 3072, device="cuda")

flux_blocks.cuda()
with torch.no_grad():
    std2 = x2.clone()
    for b in flux_blocks:
        std2 = b(std2)
    std_result2 = std2.clone()

pager2 = VRAMPager(block_size=128)
pager2.set_kernel(kernel)
for i, b in enumerate(flux_blocks):
    for name, param in b.named_parameters():
        pager2.store_weight(f"blocks.{i}.{name}", param.data)

flux_paged = nn.ModuleList([SDXLBlock(3072) for _ in range(19)]).cuda().eval()
flux_paged = replace_linear_with_paged(flux_paged, pager2)

with torch.no_grad():
    pg2 = x2.clone()
    for b in flux_paged:
        pg2 = b(pg2)
    pg_result2 = pg2.clone()

diff2 = (std_result2 - pg_result2).abs()
print(f"Max error after 19 blocks: {diff2.max().item():.6f}")
print(f"Mean error: {diff2.mean().item():.6f}")
print(f"Relative error: {(diff2 / std_result2.abs().clamp(min=1e-8)).mean().item():.6f}")
print(f"Flux-scale: {'PASS' if diff2.max().item() < 1.0 else 'FAIL'}")
print(f"Pager storage: {pager2.summary()}")

del flux_blocks, flux_paged, pager2
torch.cuda.empty_cache()

# ============================================================
# TEST 3: Wan 14B-scale model (dim=5120, 40 blocks) — Quality
# ============================================================
print()
print("=" * 60)
print("TEST 3: Wan 14B-Scale (dim=5120, 40 blocks) — Accumulated Error")
print("=" * 60)

wan_blocks = nn.ModuleList([SDXLBlock(5120) for _ in range(10)]).eval()  # 10 blocks to fit in memory
x3 = torch.randn(1, 16, 5120, device="cuda")

wan_blocks.cuda()
with torch.no_grad():
    std3 = x3.clone()
    for b in wan_blocks:
        std3 = b(std3)
    std_result3 = std3.clone()

pager3 = VRAMPager(block_size=128)
pager3.set_kernel(kernel)
for i, b in enumerate(wan_blocks):
    for name, param in b.named_parameters():
        pager3.store_weight(f"blocks.{i}.{name}", param.data)

wan_paged = nn.ModuleList([SDXLBlock(5120) for _ in range(10)]).cuda().eval()
wan_paged = replace_linear_with_paged(wan_paged, pager3)

with torch.no_grad():
    pg3 = x3.clone()
    for b in wan_paged:
        pg3 = b(pg3)
    pg_result3 = pg3.clone()

diff3 = (std_result3 - pg_result3).abs()
rel3 = (diff3 / std_result3.abs().clamp(min=1e-8))
print(f"Max error after 10 blocks: {diff3.max().item():.6f}")
print(f"Mean error: {diff3.mean().item():.6f}")
print(f"Relative error: {rel3.mean().item():.6f}")
print(f"Wan 14B-scale (10 blocks): {'PASS' if diff3.max().item() < 5.0 else 'FAIL'}")
print(f"Pager storage: {pager3.summary()}")

# Signal-to-noise ratio
snr = 20 * torch.log10(std_result3.abs().mean() / diff3.mean())
print(f"Signal-to-noise ratio: {snr.item():.1f} dB")
print(f"(>30 dB = imperceptible, >20 dB = minor, <20 dB = noticeable)")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"SDXL-scale (24 blocks, dim=1280):  max_err={diff.max().item():.4f}")
print(f"Flux-scale (19 blocks, dim=3072):  max_err={diff2.max().item():.4f}")
print(f"Wan-scale (10 blocks, dim=5120):   max_err={diff3.max().item():.4f}, SNR={snr.item():.1f}dB")
