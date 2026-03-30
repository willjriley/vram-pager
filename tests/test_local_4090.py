"""Local benchmark on RTX 4090 — PyTorch fallback path (no CUDA kernel yet)."""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import time
from pager import VRAMPager, PagedLinear

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ============================================================
print("=== TEST 1: VRAMPager Correctness ===")
pager = VRAMPager(block_size=128)

original = torch.randn(5120, 5120)  # ~100MB
pager.store_weight("test.weight", original)
print(f"Stored: {pager.summary()}")

recovered = pager.get_weight("test.weight")
diff = (original.cuda() - recovered).abs()
print(f"Max error: {diff.max().item():.6f}")
print(f"Mean error: {diff.mean().item():.6f}")
print(f"Correct (max < 0.01): {diff.max().item() < 0.01}")

del recovered, diff
torch.cuda.empty_cache()

# ============================================================
print()
print("=== TEST 2: PCIe Transfer Bandwidth (RTX 4090) ===")
SIZE = 26_000_000  # ~100MB FP32

fp32_cpu = torch.randn(SIZE, device="cpu").pin_memory()
int8_cpu = torch.randint(-128, 127, (SIZE,), dtype=torch.int8, device="cpu").pin_memory()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = fp32_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
fp32_ms = (time.time() - t0) / 50 * 1000

torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = int8_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
int8_ms = (time.time() - t0) / 50 * 1000

fp32_mb = SIZE * 4 / 1e6
int8_mb = SIZE / 1e6
print(f"FP32: {fp32_mb:.0f}MB in {fp32_ms:.1f}ms ({fp32_mb/fp32_ms*1000:.0f} MB/s)")
print(f"INT8: {int8_mb:.0f}MB in {int8_ms:.1f}ms ({int8_mb/int8_ms*1000:.0f} MB/s)")
print(f"Transfer speedup: {fp32_ms/int8_ms:.1f}x")

del fp32_cpu, int8_cpu
torch.cuda.empty_cache()

# ============================================================
print()
print("=== TEST 3: PagedLinear vs Standard Offload ===")

dims = [(5120, 5120), (5120, 13824)]

for in_dim, out_dim in dims:
    linear = nn.Linear(in_dim, out_dim, bias=False)
    cpu_weight = linear.weight.data.cpu().pin_memory()
    x = torch.randn(1, 64, in_dim, device="cuda")

    # Standard offload
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        gpu_w = cpu_weight.cuda(non_blocking=True)
        torch.cuda.synchronize()
        _ = torch.nn.functional.linear(x, gpu_w)
        torch.cuda.synchronize()
        del gpu_w
        torch.cuda.empty_cache()
    standard_ms = (time.time() - t0) / 20 * 1000

    # Paged (PyTorch fallback)
    pager = VRAMPager(block_size=128)
    pager.store_weight("layer.weight", linear.weight.data)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        w = pager.get_weight("layer.weight")
        _ = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        del w
        torch.cuda.empty_cache()
    paged_ms = (time.time() - t0) / 20 * 1000

    speedup = standard_ms / paged_ms
    print(f"[{in_dim}x{out_dim}] Standard: {standard_ms:.1f}ms | Paged: {paged_ms:.1f}ms | Speedup: {speedup:.2f}x")

    del linear, cpu_weight, pager
    torch.cuda.empty_cache()

# ============================================================
print()
print("=== PROJECTION: Wan 14B (40 layers, 20 steps) ===")
# Use the 5120x5120 timings for projection
# Each transformer block has ~4 linear layers (qkv, out, ffn_up, ffn_down)
# Estimate per-block = 4 * single_linear
# 40 blocks = 160 linear operations per step
print(f"(Based on 5120x5120 layer timings above)")
print(f"Each block has ~4 linears, 40 blocks = 160 linears/step")
# Re-measure with just one linear for clean projection
linear = nn.Linear(5120, 5120, bias=False)
cpu_w = linear.weight.data.cpu().pin_memory()
x = torch.randn(1, 64, 5120, device="cuda")

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    gw = cpu_w.cuda(non_blocking=True)
    torch.cuda.synchronize()
    del gw
standard_transfer_ms = (time.time() - t0) / 100 * 1000

pager = VRAMPager(block_size=128)
pager.store_weight("w", linear.weight.data)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    w = pager.get_weight("w")
    del w
paged_transfer_ms = (time.time() - t0) / 100 * 1000

step_std = standard_transfer_ms * 160
step_paged = paged_transfer_ms * 160
print(f"Per-layer FP32 transfer: {standard_transfer_ms:.2f}ms")
print(f"Per-layer INT8 paged:    {paged_transfer_ms:.2f}ms")
print(f"Per step (160 layers):   {step_std/1000:.1f}s standard, {step_paged/1000:.1f}s paged")
print(f"20 steps: {step_std*20/1000:.0f}s standard ({step_std*20/60000:.1f}min), {step_paged*20/1000:.0f}s paged ({step_paged*20/60000:.1f}min)")
print(f"TIME SAVED over 20 steps: {(step_std-step_paged)*20/1000:.0f}s")
print(f"SPEEDUP on transfer: {standard_transfer_ms/paged_transfer_ms:.2f}x")
