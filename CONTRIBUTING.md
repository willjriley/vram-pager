# Contributing to VRAM Pager

Thanks for your interest in contributing! This project brings compressed GPU memory paging to the diffusion/video model community.

## How to Help

### Testing
- Test on different GPUs (RTX 3060, 3070, 3080, 4060, 4070, 4080)
- Test with different models (SDXL, Flux, Hunyuan, CogVideo)
- Report benchmarks: step time with and without VRAM Pager

### Code
- Linux `.so` kernel builds
- INT4 dequantization kernel
- AMD ROCm kernel port
- Apple Metal kernel port
- SSD three-tier paging
- Better ComfyUI integration (GGUF model support)

### Documentation
- Installation guides for different OS/GPU combos
- Tutorial videos
- Benchmark comparisons

## Development Setup

```bash
git clone https://github.com/willjriley/vram-pager.git
cd vram-pager

# Build CUDA kernel (requires CUDA Toolkit 12.x)
cd build
nvcc -O2 --shared -gencode=arch=compute_89,code=sm_89 \
  -Xcompiler /LD -o dequant.dll dequant.cu -lcudart

# Run tests
python tests/test_local_4090.py
python tests/test_real_model.py
```

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Ensure existing tests pass
5. Submit PR with clear description

## Code Style

- Python: Follow existing patterns, type hints appreciated
- CUDA: Keep kernels minimal and well-commented
- Avoid unnecessary dependencies
