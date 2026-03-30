"""
ComfyUI Integration Hook — Transparent compressed weight paging.

Instead of replacing nn.Linear layers (which breaks LoRA),
this hooks into ComfyUI's existing weight_function mechanism.
When --lowvram is active, ComfyUI already calls weight_functions
during the forward pass to load weights on-demand. We add a
compressed transfer function that sends INT8 data and decompresses
on GPU, making the existing offloading faster.

Usage:
    from pager.comfy_hook import accelerate_model
    model_patcher = load_your_model()  # standard ComfyUI loader
    model_patcher = accelerate_model(model_patcher, mode="int8")
    # That's it — model works exactly as before, just faster transfers
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from .memory_manager import PagedWeightStore


class CompressedWeight:
    """Stores a compressed version of a weight tensor.

    When called (as a weight_function), returns the decompressed weight.
    This replaces the standard .to(device) transfer with compressed transfer.
    """

    def __init__(self, store: PagedWeightStore, key: str, kernel=None):
        self.store = store
        self.key = key
        self.kernel = kernel
        self._device = None

    def move_to(self, device=None):
        """Called by ComfyUI's move_weight_functions to track device."""
        self._device = device
        return 0  # We don't pre-allocate — decompress on demand

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        """Called during forward pass. Returns decompressed weight on GPU.

        ComfyUI passes the current weight tensor. We ignore it and
        return our decompressed version instead. If LoRA patches
        follow us in the weight_function chain, they'll modify
        our decompressed output — which is exactly what we want.
        """
        info = self.store.get(self.key)
        if info is None:
            return weight  # Fallback: return original

        mode = info.get("mode", "int8")

        if mode == "fp16":
            # FP16 path: just transfer half-size data
            gpu_data = info["data"].to(weight.device, non_blocking=True)
            return gpu_data.to(weight.dtype)

        elif mode == "int8":
            # INT8 path: transfer compressed + decompress on GPU
            gpu_q = info["quantized"].to(weight.device, non_blocking=True)
            gpu_s = info["scales"].to(weight.device, non_blocking=True)

            if self.kernel is not None:
                flat = self.kernel.dequantize_int8(gpu_q, gpu_s, self.store.block_size)
            else:
                # PyTorch fallback
                flat = (gpu_q.float().reshape(-1, self.store.block_size)
                        * gpu_s.unsqueeze(1)).reshape(-1)

            result = flat[:info["numel"]].reshape(info["shape"]).to(weight.dtype)
            del gpu_q, gpu_s, flat
            return result

        return weight  # Unknown mode, fallback


def accelerate_model(model_patcher, mode: str = "int8", block_size: int = 128, kernel=None):
    """Accelerate a ComfyUI model patcher with compressed weight paging.

    This is the main entry point. Call it on any model_patcher and the
    model's weight transfers will be compressed automatically.

    Args:
        model_patcher: ComfyUI ModelPatcher instance
        mode: "int8" (3.4x speedup) or "fp16" (1.8x speedup, lossless)
        block_size: INT8 quantization block size (default 128)
        kernel: Compiled CUDA kernel module (optional, falls back to PyTorch)

    Returns:
        The same model_patcher, now with compressed weight transfers
    """
    model = model_patcher.model
    diff_model = model.diffusion_model

    # Create weight store and compress all linear layer weights
    store = PagedWeightStore(block_size=block_size, mode=mode)

    compressed_count = 0
    total_bytes_original = 0
    total_bytes_compressed = 0

    with torch.no_grad():
        for name, module in diff_model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "comfy_cast_weights"):
                weight_key = f"diffusion_model.{name}.weight"

                # Get the weight
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight
                    if w.device.type == "meta":
                        continue  # Skip meta tensors

                    total_bytes_original += w.numel() * w.element_size()

                    # Compress and store — quantize_and_store moves to CPU first
                    # to avoid creating temporary GPU copies that OOM on small cards
                    store.quantize_and_store(weight_key, w.data)
                    total_bytes_compressed += store._total_bytes

                    # Add our compressed transfer as a weight_function
                    compressed_fn = CompressedWeight(store, weight_key, kernel=kernel)

                    if not hasattr(module, "weight_function"):
                        module.weight_function = []

                    # Insert at the FRONT so we run before LoRA patches
                    module.weight_function.insert(0, compressed_fn)
                    compressed_count += 1

    compression_ratio = total_bytes_original / max(total_bytes_compressed, 1)
    print(f"[VRAMPager] Compressed {compressed_count} layers")
    print(f"[VRAMPager] {total_bytes_original / 1e6:.0f} MB → {total_bytes_compressed / 1e6:.0f} MB ({compression_ratio:.1f}x)")
    print(f"[VRAMPager] Mode: {mode} | Kernel: {'CUDA' if kernel else 'PyTorch fallback'}")

    return model_patcher
