# Prior Art — Compressed GPU Memory Paging

## Existing Solutions

### FlexGen (Stanford/UC Berkeley, 2023)
- **Paper**: "High-Throughput Generative Inference of Large Language Models with a Single GPU"
- **What it does**: Offloads to GPU/CPU/disk with a linear programming optimizer that finds optimal tensor placement
- **Compression**: Group-wise INT4 quantization with GPU-side decompression
- **Key insight**: Overlaps computation, memory transfer, and disk I/O
- **GPU decompression**: YES — FlexGen does decompress on GPU, same core technique as us
- **Limitation**: LLM-only, batch throughput focused, not integrated with diffusion frameworks
- **Relevance**: Closest prior art. Our differentiation is application domain (diffusion/video) and packaging (drop-in PyTorch module + ComfyUI integration), NOT the decompression technique itself

### PowerInfer (SJTU, 2024)
- **What it does**: Exploits "hot neuron" locality — most neurons are rarely activated
- **Approach**: Keep hot neurons (top ~10%) in VRAM, page cold neurons from CPU
- **Key insight**: Transformer attention patterns are sparse — not all weights are needed every forward pass
- **Limitation**: Requires activation profiling per model, LLM-specific
- **Relevance**: The hot/cold partitioning concept could apply to diffusion models (early layers vs late layers have different memory patterns)

### llama.cpp
- **What it does**: Quantized inference with optional GPU offload
- **Compression**: Custom GGUF format with per-block quantization (Q2_K through Q8_0)
- **Offloading**: Can split layers between GPU and CPU
- **Limitation**: Doesn't decompress on GPU — dequantizes on CPU then transfers FP16
- **Relevance**: GGUF quantization format is well-tested, could reuse the format

### bitsandbytes (HuggingFace)
- **What it does**: INT8/INT4 inference for PyTorch models
- **Approach**: Quantized weights stored ON GPU, dequantized per-matmul
- **Key insight**: 4-bit NormalFloat (NF4) maintains quality better than uniform INT4
- **Limitation**: Assumes model fits in VRAM (no paging). If it doesn't fit, you're stuck.
- **Relevance**: The NF4 quantization scheme is proven and could be used for our compressed format

### ComfyUI --lowvram
- **What it does**: Moves entire model layers between CPU and GPU
- **Approach**: Before computing layer N, move it to GPU. After, move it back.
- **Compression**: None — transfers full FP16/FP32
- **Limitation**: Bus transfer is the bottleneck. No overlap, no compression.
- **Relevance**: This is our baseline. We need to beat ~42 sec/step for Wan 2.1 14B.

### NVIDIA Unified Virtual Memory (UVM)
- **What it does**: Hardware-level page faulting between GPU and CPU memory
- **Approach**: GPU accesses a virtual address, if not in VRAM, hardware pages it from system RAM
- **Limitation**: Page fault latency is high (~10-50 microseconds), no compression
- **Relevance**: Shows that NVIDIA considers this problem important enough for hardware support

## Gap Analysis

| Feature | FlexGen | PowerInfer | llama.cpp | bitsandbytes | --lowvram | **Us** |
|---------|---------|------------|-----------|--------------|-----------|--------|
| Compressed transfer | INT4 | No | No | N/A | No | **INT4/INT8/FP16** |
| GPU decompression | **Yes** | No | **Yes** | Yes | No | **Yes** |
| Async prefetch | Yes | Yes | No | N/A | No | **Yes** |
| Diffusion model support | No | No | No | Partial | Yes | **Yes** |
| Video model support | No | No | No | No | Yes | **Yes** |
| ComfyUI integration | No | No | Via GGUF | No | Native | **Yes** |
| Drop-in nn.Linear | No | No | No | Yes | No | **Yes** |

**Our unique position**: The core technique (compressed transfer + GPU decompression) exists in the LLM world via FlexGen and llama.cpp. Nobody has built it for diffusion/video models with ComfyUI/PyTorch integration. That's the gap we fill.

## Technical References

- FlexGen paper: https://arxiv.org/abs/2303.06865
- PowerInfer paper: https://arxiv.org/abs/2312.12456
- bitsandbytes NF4: https://arxiv.org/abs/2305.14314
- CUDA pinned memory: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pinned-memory
- CUDA streams (async): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
