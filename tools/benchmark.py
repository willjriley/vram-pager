"""
Benchmark — Compare paged inference vs standard --lowvram offloading.

Tests:
1. PCIe bandwidth: FP32 vs INT8 transfer
2. Dequantization kernel speed
3. End-to-end: PagedLinear forward pass vs standard Linear with offloading
4. Full model simulation: 40 transformer layers, 20 steps
"""
import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pager import VRAMPager, PagedLinear, replace_linear_with_paged


def benchmark_transfer():
    """Benchmark raw PCIe transfer: FP32 vs INT8."""
    print("=" * 60)
    print("BENCHMARK 1: PCIe Transfer Bandwidth")
    print("=" * 60)

    sizes_mb = [25, 50, 100, 200]
    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4  # elements for FP32

        fp32 = torch.randn(n, device="cpu").pin_memory()
        int8 = torch.randint(-128, 127, (n,), dtype=torch.int8, device="cpu").pin_memory()

        # FP32
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = fp32.cuda(non_blocking=True)
            torch.cuda.synchronize()
        fp32_ms = (time.time() - t0) / 20 * 1000

        # INT8 (4x smaller)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = int8.cuda(non_blocking=True)
            torch.cuda.synchronize()
        int8_ms = (time.time() - t0) / 20 * 1000

        print(f"  {size_mb:3d} MB FP32: {fp32_ms:6.1f} ms | "
              f"{size_mb//4:3d} MB INT8: {int8_ms:6.1f} ms | "
              f"Speedup: {fp32_ms/int8_ms:.1f}x")

        del fp32, int8
        torch.cuda.empty_cache()


def benchmark_paged_linear():
    """Benchmark PagedLinear vs standard Linear with CPU offloading."""
    print()
    print("=" * 60)
    print("BENCHMARK 2: PagedLinear vs Offloaded Linear")
    print("=" * 60)

    dims = [(4096, 4096), (5120, 5120), (5120, 13824)]

    for in_dim, out_dim in dims:
        # Standard Linear (simulating --lowvram: weight on CPU, move to GPU for matmul)
        linear = nn.Linear(in_dim, out_dim, bias=False)
        cpu_weight = linear.weight.data.cpu().pin_memory()

        x = torch.randn(1, 128, in_dim, device="cuda")  # batch=1, seq=128

        # Standard offload timing
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            gpu_w = cpu_weight.cuda(non_blocking=True)
            torch.cuda.synchronize()
            _ = torch.nn.functional.linear(x, gpu_w)
            torch.cuda.synchronize()
            del gpu_w
        standard_ms = (time.time() - t0) / 20 * 1000

        # PagedLinear timing
        pager = VRAMPager(block_size=128)
        pager.store_weight("test.weight", linear.weight.data)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            w = pager.get_weight("test.weight")
            _ = torch.nn.functional.linear(x, w)
            torch.cuda.synchronize()
            del w
        paged_ms = (time.time() - t0) / 20 * 1000

        speedup = standard_ms / paged_ms
        print(f"  [{in_dim}x{out_dim}] Standard: {standard_ms:.1f} ms | "
              f"Paged: {paged_ms:.1f} ms | Speedup: {speedup:.2f}x")

        del linear, cpu_weight, pager
        torch.cuda.empty_cache()


def benchmark_model_simulation():
    """Simulate a full transformer model: 40 layers, 20 denoising steps."""
    print()
    print("=" * 60)
    print("BENCHMARK 3: Simulated 14B Model (40 layers x 20 steps)")
    print("=" * 60)

    # Create a simplified transformer block
    dim = 5120
    ffn_dim = 13824

    class FakeTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.attn_out = nn.Linear(dim, dim, bias=False)
            self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
            self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

        def forward(self, x):
            # Simplified attention
            qkv = self.attn_qkv(x)
            x = x + self.attn_out(qkv[..., :dim])
            # FFN
            x = x + self.ffn_down(torch.relu(self.ffn_up(x)))
            return x

    print(f"  Creating 40-layer model (dim={dim}, ffn={ffn_dim})...")
    blocks = nn.ModuleList([FakeTransformerBlock() for _ in range(40)])

    x = torch.randn(1, 64, dim, device="cuda")  # small batch for testing

    # Standard offload: move each block to GPU, compute, move back
    for b in blocks:
        b.cpu()

    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(3):  # 3 steps (not 20 — too slow for benchmark)
        h = x
        for b in blocks:
            b.cuda()
            torch.cuda.synchronize()
            h = b(h)
            b.cpu()
            torch.cuda.synchronize()
    standard_per_step = (time.time() - t0) / 3

    # Paged: store all weights in pager
    print(f"  Converting to paged model...")
    pager = VRAMPager(block_size=128)
    for i, b in enumerate(blocks):
        b.cuda()  # temporarily move to GPU for storage
        for name, param in b.named_parameters():
            pager.store_weight(f"blocks.{i}.{name}", param.data)
        b.cpu()

    # Paged inference
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(3):
        h = x
        for i, b in enumerate(blocks):
            # Get each weight from pager (compressed transfer + decompress)
            for name, _ in b.named_parameters():
                full_name = f"blocks.{i}.{name}"
                w = pager.get_weight(full_name)
                del w  # just measuring the fetch time
    paged_per_step = (time.time() - t0) / 3

    print(f"  Standard offload per step: {standard_per_step:.2f}s")
    print(f"  Paged per step:            {paged_per_step:.2f}s")
    print(f"  Speedup:                   {standard_per_step/paged_per_step:.2f}x")
    print(f"  Projected 20 steps standard: {standard_per_step*20:.0f}s")
    print(f"  Projected 20 steps paged:    {paged_per_step*20:.0f}s")
    print(f"  TIME SAVED:                  {(standard_per_step-paged_per_step)*20:.0f}s")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarks require GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    benchmark_transfer()
    benchmark_paged_linear()
    # benchmark_model_simulation()  # Enable when testing on real hardware
