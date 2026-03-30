---
date: 2026-03-30
hardware: NVIDIA L40S (RunPod)
cuda: 12.4
arch: sm_89 (Ada Lovelace — same as RTX 4090)
---

# VRAM Pager Kernel Benchmark Results

## INT8 Dequantization Kernel
- **Elements**: 10,000,000
- **CUDA kernel**: 0.055 ms
- **PyTorch equivalent**: 0.081 ms
- **Speedup**: 1.5x (kernel overhead is negligible)
- **Correctness**: Max error = 0.00000000 (perfect)
- **Block size**: 128 elements per scale factor

## PCIe Transfer Bandwidth
- **FP32**: 104 MB in 4.2 ms (25.0 GB/s)
- **INT8**: 26 MB in 1.0 ms (25.6 GB/s)
- **Transfer speedup**: 4.1x (4x less data at same bandwidth)
- **INT8 transfer + decompress**: 1.1 ms total
- **Net speedup vs FP32**: 3.9x

## Projected Impact on Wan 2.1 14B (40 layers)
- Per step FP32 offload: 0.2s (just transfer, not compute)
- Per step INT8 paged: 0.04s
- **74% reduction in transfer overhead**

## Note on Real-World Impact
The L40S has very fast PCIe (Gen5). On the RTX 4090 (PCIe Gen4 x16, ~25 GB/s theoretical):
- The absolute transfer times will be similar
- BUT the 4090 has 16GB VRAM vs L40S 48GB
- On 16GB, MORE layers need to be offloaded per step
- So the 3.9x transfer speedup applies to a LARGER portion of the step time
- Estimated real-world impact on 4090: 40-60% step time reduction for offloaded models

## Compilation
- Compiled via `torch.utils.cpp_extension.load()` with `nvcc` and `ninja`
- Compile time: 54.3 seconds
- Target arch: sm_89 (compute_89)
- Output: `vram_pager_v2.so` (196 KB)
