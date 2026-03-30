"""Upload kernel code to RunPod, compile, and benchmark."""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.runpod_exec import run

# Step 1: Write the kernel source file on the pod
run("""
import os
os.makedirs("/workspace/vram_pager", exist_ok=True)

with open("/workspace/vram_pager/kernels.cu", "w") as f:
    f.write(r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dequant_int8_kernel(
    const int8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int num_elements,
    int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int scale_idx = i / block_size;
        output[i] = (float)quantized[i] * scales[scale_idx];
    }
}

__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    float* __restrict__ output,
    int num_elements,
    int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = i * 2;
    if (elem_idx < num_elements) {
        uint8_t packed = quantized[i];
        int scale_idx = elem_idx / block_size;
        float scale = scales[scale_idx];
        float zero = zeros[scale_idx];
        output[elem_idx] = ((float)(packed & 0x0F) - zero) * scale;
        if (elem_idx + 1 < num_elements)
            output[elem_idx + 1] = ((float)((packed >> 4) & 0x0F) - zero) * scale;
    }
}

torch::Tensor dequantize_int8(torch::Tensor quantized, torch::Tensor scales, int block_size) {
    auto n = quantized.numel();
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8_kernel<<<blocks, threads>>>(
        quantized.data_ptr<int8_t>(), scales.data_ptr<float>(),
        output.data_ptr<float>(), n, block_size);
    return output;
}

torch::Tensor dequantize_int4(torch::Tensor quantized, torch::Tensor scales, torch::Tensor zeros, int block_size) {
    auto n_packed = quantized.numel();
    auto n_elements = n_packed * 2;
    auto output = torch::empty({n_elements}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks_grid = (n_packed + threads - 1) / threads;
    dequant_int4_kernel<<<blocks_grid, threads>>>(
        quantized.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros.data_ptr<float>(),
        output.data_ptr<float>(), n_elements, block_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_int8", &dequantize_int8, "INT8 dequantization");
    m.def("dequantize_int4", &dequantize_int4, "INT4 dequantization");
}
''')
print("Kernel source written!")
""", timeout=30)

# Step 2: Compile with torch cpp_extension
print("\n--- COMPILING ---")
run("""
import torch
from torch.utils.cpp_extension import load
import time

print("Compiling CUDA kernel (this takes ~30-60 seconds first time)...")
t0 = time.time()
vram_pager = load(
    name="vram_pager_kernel",
    sources=["/workspace/vram_pager/kernels.cu"],
    verbose=True
)
print(f"COMPILED in {time.time()-t0:.1f}s!")
print(f"Module: {vram_pager}")
print(f"Functions: {dir(vram_pager)}")
""", timeout=120)

# Step 3: Benchmark
print("\n--- BENCHMARKING ---")
run("""
import torch
from torch.utils.cpp_extension import load
import time

vram_pager = load(name="vram_pager_kernel", sources=["/workspace/vram_pager/kernels.cu"])

# ============ INT8 TEST ============
print("=== INT8 DEQUANTIZATION BENCHMARK ===")
N = 10_000_000
BLOCK_SIZE = 128

original = torch.randn(N, device="cuda")
scales = original.reshape(-1, BLOCK_SIZE).abs().max(dim=1).values / 127.0
quantized = (original.reshape(-1, BLOCK_SIZE) / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8).reshape(-1)

# Warmup
for _ in range(10):
    _ = vram_pager.dequantize_int8(quantized, scales, BLOCK_SIZE)
torch.cuda.synchronize()

# Benchmark kernel
t0 = time.time()
for _ in range(1000):
    output = vram_pager.dequantize_int8(quantized, scales, BLOCK_SIZE)
torch.cuda.synchronize()
kernel_ms = (time.time() - t0) / 1000 * 1000

# Benchmark PyTorch equivalent
for _ in range(10):
    _ = quantized.float().reshape(-1, BLOCK_SIZE) * scales.unsqueeze(1)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    pt_out = (quantized.float().reshape(-1, BLOCK_SIZE) * scales.unsqueeze(1)).reshape(-1)
torch.cuda.synchronize()
pytorch_ms = (time.time() - t0) / 1000 * 1000

max_diff = (output - pt_out).abs().max().item()
print(f"Elements:     {N:,}")
print(f"CUDA kernel:  {kernel_ms:.3f} ms")
print(f"PyTorch:      {pytorch_ms:.3f} ms")
print(f"Speedup:      {pytorch_ms/kernel_ms:.1f}x")
print(f"Max diff:     {max_diff:.8f}")
print(f"Correct:      {max_diff < 1e-5}")

# ============ PCIE BANDWIDTH ============
print()
print("=== PCIe BANDWIDTH: COMPRESSED vs UNCOMPRESSED ===")
SIZE_ELEMENTS = 26_000_000  # ~100MB FP32 = one transformer layer

fp32_cpu = torch.randn(SIZE_ELEMENTS, device="cpu").pin_memory()
int8_cpu = torch.randint(-128, 127, (SIZE_ELEMENTS,), dtype=torch.int8, device="cpu").pin_memory()

# FP32 transfer
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = fp32_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
fp32_ms = (time.time() - t0) / 50 * 1000

# INT8 transfer (4x smaller)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = int8_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
int8_ms = (time.time() - t0) / 50 * 1000

fp32_mb = SIZE_ELEMENTS * 4 / 1e6
int8_mb = SIZE_ELEMENTS * 1 / 1e6

print(f"FP32 transfer: {fp32_mb:.0f} MB in {fp32_ms:.1f} ms ({fp32_mb/fp32_ms*1000:.0f} MB/s)")
print(f"INT8 transfer: {int8_mb:.0f} MB in {int8_ms:.1f} ms ({int8_mb/int8_ms*1000:.0f} MB/s)")
print(f"Transfer speedup: {fp32_ms/int8_ms:.1f}x")
print(f"INT8 transfer + decompress: {int8_ms + kernel_ms:.1f} ms")
print(f"NET SPEEDUP vs FP32: {fp32_ms/(int8_ms + kernel_ms):.1f}x")
print()
print(f"=== PROJECTED IMPACT ON WAN 2.1 14B (40 layers) ===")
per_layer_fp32 = fp32_ms
per_layer_int8 = int8_ms + kernel_ms
total_fp32 = per_layer_fp32 * 40
total_int8 = per_layer_int8 * 40
print(f"FP32 offload per step: {total_fp32/1000:.1f}s")
print(f"INT8 paged per step:   {total_int8/1000:.1f}s")
print(f"Savings per step:      {(total_fp32-total_int8)/1000:.1f}s")
print(f"Over 20 steps:         {(total_fp32-total_int8)*20/1000:.0f}s saved")
""", timeout=120)
