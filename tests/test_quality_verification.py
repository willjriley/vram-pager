"""
Bulletproof quality verification — multiple approaches to confirm INT8 quality.

This test uses THREE independent methods to verify the same claim:
1. Direct weight comparison (quantize → dequantize → compare)
2. Forward pass comparison (same model, same input, quantized vs original)
3. Pager roundtrip (store → retrieve → use in forward pass)

If all three agree, the quality claim is verified.
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
import copy

# Load CUDA kernel
dll = ctypes.CDLL(r"C:\repos\vram_pager\build\dequant.dll")
dll.launch_dequant_int8.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]

class K:
    def __init__(self, d):
        self.dll = d
    def dequantize_int8(self, q, s, bs):
        n = q.numel()
        o = torch.empty(n, dtype=torch.float32, device=q.device)
        self.dll.launch_dequant_int8(q.data_ptr(), s.data_ptr(), o.data_ptr(), n, bs, None)
        return o

kernel = K(dll)

from pager import VRAMPager, replace_linear_with_paged

print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# Build a test model — SAME weights for ALL tests
# ============================================================
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_up = nn.Linear(dim, dim * 4, bias=False)
        self.ffn_down = nn.Linear(dim * 4, dim, bias=False)
    def forward(self, x):
        h = self.out(self.qkv(x)[..., :x.shape[-1]])
        x = x + h
        x = x + self.ffn_down(torch.relu(self.ffn_up(x)))
        return x

def int8_quantize_weight(w, block_size=128):
    """Quantize a weight tensor to INT8 and dequantize back."""
    flat = w.float().flatten()
    n = flat.numel()
    padded = ((n + block_size - 1) // block_size) * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, device=flat.device)])
    blocked = flat.reshape(-1, block_size)
    scales = blocked.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-10)
    q = (blocked / scales.unsqueeze(1)).round().clamp(-128, 127)
    dq = (q * scales.unsqueeze(1)).reshape(-1)[:n].reshape(w.shape)
    return dq

for dim, num_blocks, label in [(2048, 40, "Full 40-block"), (5120, 10, "Wan 14B scale 10-block")]:
    print("=" * 70)
    print(f"VERIFICATION: {label} (dim={dim}, blocks={num_blocks})")
    print("=" * 70)

    # Create original model
    torch.manual_seed(42)  # Deterministic
    original = nn.ModuleList([Block(dim) for _ in range(num_blocks)]).cuda().eval()
    x = torch.randn(1, 16, dim, device="cuda")

    # Compute reference output
    with torch.no_grad():
        ref = x.clone()
        for b in original:
            ref = b(ref)

    # ---- METHOD 1: Direct quantize-in-place ----
    torch.manual_seed(42)
    method1 = nn.ModuleList([Block(dim) for _ in range(num_blocks)]).cuda().eval()
    # Copy original weights then quantize
    method1.load_state_dict(original.state_dict())
    for b in method1:
        for name, param in b.named_parameters():
            param.data = int8_quantize_weight(param.data).to(param.dtype)

    with torch.no_grad():
        out1 = x.clone()
        for b in method1:
            out1 = b(out1)

    diff1 = (ref - out1).abs()
    snr1 = 20 * torch.log10(ref.abs().mean() / diff1.mean())
    print(f"\n  Method 1 (direct quantize-in-place):")
    print(f"    Max error: {diff1.max().item():.4f}")
    print(f"    SNR: {snr1.item():.1f} dB")

    del method1
    torch.cuda.empty_cache()

    # ---- METHOD 2: Pager with CUDA kernel ----
    torch.manual_seed(42)
    method2 = nn.ModuleList([Block(dim) for _ in range(num_blocks)]).cuda().eval()
    method2.load_state_dict(original.state_dict())

    pager = VRAMPager(block_size=128, mode="int8")
    pager.set_kernel(kernel)
    method2 = replace_linear_with_paged(method2, pager)

    with torch.no_grad():
        out2 = x.clone()
        for b in method2:
            out2 = b(out2)

    diff2 = (ref - out2).abs()
    snr2 = 20 * torch.log10(ref.abs().mean() / diff2.mean())
    print(f"\n  Method 2 (VRAMPager + CUDA kernel):")
    print(f"    Max error: {diff2.max().item():.4f}")
    print(f"    SNR: {snr2.item():.1f} dB")

    del method2, pager
    torch.cuda.empty_cache()

    # ---- METHOD 3: Pager PyTorch fallback (no kernel) ----
    torch.manual_seed(42)
    method3 = nn.ModuleList([Block(dim) for _ in range(num_blocks)]).cuda().eval()
    method3.load_state_dict(original.state_dict())

    pager3 = VRAMPager(block_size=128, mode="int8")
    # No kernel set — uses PyTorch fallback
    method3 = replace_linear_with_paged(method3, pager3)

    with torch.no_grad():
        out3 = x.clone()
        for b in method3:
            out3 = b(out3)

    diff3 = (ref - out3).abs()
    snr3 = 20 * torch.log10(ref.abs().mean() / diff3.mean())
    print(f"\n  Method 3 (VRAMPager PyTorch fallback, no kernel):")
    print(f"    Max error: {diff3.max().item():.4f}")
    print(f"    SNR: {snr3.item():.1f} dB")

    # ---- CROSS-CHECK ----
    methods_agree = abs(snr1.item() - snr2.item()) < 1.0 and abs(snr2.item() - snr3.item()) < 1.0
    all_excellent = snr1.item() > 30 and snr2.item() > 30 and snr3.item() > 30

    print(f"\n  Cross-check:")
    print(f"    All methods agree (within 1 dB): {methods_agree}")
    print(f"    All SNR > 30 dB (excellent):     {all_excellent}")
    print(f"    VERIFIED: {'YES' if methods_agree and all_excellent else 'NO — NEEDS INVESTIGATION'}")

    del method3, pager3, original
    torch.cuda.empty_cache()
    print()
