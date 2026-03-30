# VRAM Pager — TODO

## Completed

- [x] CUDA INT8 decompression kernel (compiled Windows + Linux)
- [x] FP16 lossless paging mode (bit-perfect, 1.8x speedup)
- [x] Memory manager with pinned memory + async CUDA stream prefetch
- [x] PagedLinear drop-in nn.Linear replacement
- [x] ComfyUI plugin (PagedModelLoader + PagedModelLoaderGGUF nodes)
- [x] Multi-GPU verified (RTX 4090 Windows, A6000 Linux, L40S Linux)
- [x] Quality triple-verified (37-43 dB SNR, 3 independent methods)
- [x] Red-teamed by independent AI — all claims verified
- [x] Wan 2.2 14B full render completed (5.3 sec/step vs 448 sec/step)
- [x] Pre-compiled kernel for RTX 40-series (sm_89 dequant.dll included)

## Priority 1: Before Public Release

- [x] **LoRA compatibility** — Refactored to hook-based approach (CompressedPager node). Model structure untouched, LoRAs work normally. Tested with Snake Eyes, Cobra Commander, Serpentor LoRAs — proper face identity confirmed.
- [x] **Test with SDXL end-to-end** — Tested via CheckpointLoaderSimple + CompressedPager + LoRA. Generated images with Snake Eyes, Cobra Commander, Serpentor LoRAs — full pipeline working.
- [ ] **Test with Flux end-to-end** — Not yet tested, not a release blocker
- [x] **GGUF compatibility documented** — GGUF models don't benefit (already quantized). Documented in README.
- [x] **Side-by-side visual comparison** — Comparison image and video created (assets/)
- [x] **Pre-compiled kernel for RTX 30-series** — sm_86 .so compiled, sm_80 .so compiled

## Priority 2: Wider Adoption

- [ ] **INT4 kernel** — 8:1 compression, ~7x transfer speedup
- [x] **pip installable** — pyproject.toml + wheel built (dist/vram_pager-0.1.0-py3-none-any.whl)
- [ ] **Pre-compiled kernels for all architectures** — sm_80 (A100), sm_90 (H100)
- [ ] **Benchmark vs GGUF Q4** — Fair comparison against common quantized format
- [ ] **Error handling** — Graceful fallback with clear messages when kernel fails
- [ ] **ComfyUI Manager listing** — One-click install from ComfyUI Manager

## Priority 3: Features

- [ ] **SSD three-tier paging** — NVMe -> RAM -> GPU for models exceeding RAM
- [ ] **Adaptive per-step precision** — INT4 for noisy steps, FP16 for final clean steps
- [ ] **Per-layer precision** — Attention at FP16, FFN at INT4
- [ ] **LLM support** — Token generation with paged KV cache
- [ ] **AMD ROCm kernel** — HIP port for AMD GPUs
- [ ] **Apple Metal kernel** — Port for M-series Macs
- [ ] **Hugging Face Diffusers integration**
- [ ] **InvokeAI integration**

## Priority 3.5: Community-Requested Benchmarks

- [ ] **Benchmark vs ComfyUI dynamic VRAM** — `--fast dynamic_vram` (aimdo). Test if they're complementary or redundant. Requires installing `comfy-aimdo` package.
- [ ] **Test stacking with sage attention / torch.compile** — Verify compressed paging + compute optimizations stack without conflict
- [ ] **Test on 8GB VRAM** (RTX 3060 / 4060) — Community interest in low-VRAM testing

## Priority 4: Polish

- [ ] **Demo video** — Screen recording of before/after speed comparison
- [ ] **GitHub Actions CI** — Automated testing
- [ ] **Documentation site**
- [ ] **PyPI release**

## Multi-GPU Test Status

| GPU | Architecture | Speedup | Platform | Status |
|-----|-------------|---------|----------|--------|
| RTX 4090 | sm_89 (Ada) | 3.42x | Windows | Done |
| RTX A6000 | sm_86 (Ampere) | 3.46x | Linux | Done |
| NVIDIA L40S | sm_89 (Ada) | 2.88x | Linux | Done |
| RTX 3090 | sm_86 (Ampere) | — | — | Not tested |
| A100 | sm_80 (Ampere) | — | — | Not tested |
| H100 | sm_90 (Hopper) | — | — | Not tested |
