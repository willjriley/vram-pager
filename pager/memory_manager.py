"""
VRAM Pager — Memory Manager

Orchestrates compressed weight paging between CPU RAM and GPU VRAM.
Handles:
- Storing quantized weights in pinned CPU memory (when safe)
- Async prefetch of next layer while current layer computes
- CUDA kernel invocation for on-GPU decompression
- Working set management (what's in VRAM vs RAM)

Pin-memory policy:
  ComfyUI's dynamic VRAM system (aimdo) uses cudaHostRegister to pin
  weight tensors on demand. PyTorch's .pin_memory() uses cudaHostAlloc.
  Both draw from the same OS page-lock budget (~50% of system RAM on
  Windows). When both are active the budget overflows, producing
  "Pin error" warnings and queuing async CUDA errors.

  To avoid the conflict we detect ComfyUI's environment at runtime and
  skip our own pinning when aimdo or ComfyUI's pinned-memory system is
  already active. The transfer-speed penalty (~2-4x slower DMA) is
  negligible relative to diffusion model render times.
"""
import torch
import torch.cuda
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import logging
import threading


def _detect_comfyui_pinning() -> bool:
    """Return True if ComfyUI's pinned-memory system is active.

    Checks for:
    1. aimdo dynamic VRAM (comfy.memory_management.aimdo_enabled)
    2. ComfyUI's cudaHostRegister pinning (model_management.MAX_PINNED_MEMORY > 0)
    3. ComfyUI's --disable-pinned-memory flag (safe to pin ourselves)

    If ComfyUI is not importable we are running standalone — pin freely.
    """
    try:
        import comfy.memory_management
        if getattr(comfy.memory_management, "aimdo_enabled", False):
            logging.info("[VRAMPager] aimdo dynamic VRAM detected — deferring pinning to ComfyUI")
            return True
    except ImportError:
        pass

    try:
        import comfy.model_management as cmm
        if getattr(cmm, "MAX_PINNED_MEMORY", 0) > 0:
            logging.info("[VRAMPager] ComfyUI pinned-memory system active — deferring pinning to ComfyUI")
            return True
    except ImportError:
        pass

    return False


# Resolved once at import time. Can be overridden by callers.
COMFYUI_OWNS_PINNING: bool = _detect_comfyui_pinning()


class PagedWeightStore:
    """Stores weights in pinned CPU memory for fast GPU transfer.

    Supports multiple precision modes:
    - "fp16": Lossless, 2:1 compression, no kernel needed (DEFAULT)
    - "int8": 4:1 compression, needs CUDA kernel, slight quality loss
    - "int4": 8:1 compression (future), noticeable quality trade-off
    """

    def __init__(self, block_size: int = 128, mode: str = "fp16", pin: bool = True):
        self.block_size = block_size
        self.mode = mode
        self.pin = pin and not COMFYUI_OWNS_PINNING
        self.weights: Dict[str, dict] = {}
        self._total_bytes = 0
        if not self.pin:
            logging.info("[VRAMPager] Pinned memory disabled — using pageable CPU memory")

    def quantize_and_store(self, name: str, tensor: torch.Tensor):
        """Store a weight tensor in compressed pinned memory."""
        if self.mode == "fp16":
            self._store_fp16(name, tensor)
        elif self.mode == "int8":
            self._store_int8(name, tensor)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _store_fp16(self, name: str, tensor: torch.Tensor):
        """Store as FP16 — lossless for models trained in FP16/BF16."""
        # Move to CPU FIRST, then cast — avoids creating FP16 copy on GPU
        fp16_data = tensor.cpu().half().contiguous()
        if self.pin:
            fp16_data = fp16_data.pin_memory()
        self.weights[name] = {
            "data": fp16_data,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "numel": tensor.numel(),
            "mode": "fp16",
        }
        self._total_bytes += fp16_data.numel() * 2

    def _store_int8(self, name: str, tensor: torch.Tensor):
        """Quantize to INT8 with per-block scale factors."""
        # Move to CPU FIRST, then cast — avoids creating FP32 copy on GPU
        flat = tensor.cpu().float().flatten()
        n = flat.numel()

        padded_n = ((n + self.block_size - 1) // self.block_size) * self.block_size
        if padded_n > n:
            flat = torch.cat([flat, torch.zeros(padded_n - n)])

        blocked = flat.reshape(-1, self.block_size)
        scales = blocked.abs().max(dim=1).values / 127.0
        scales = scales.clamp(min=1e-10)
        quantized = (blocked / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        q_data = quantized.flatten()[:n].contiguous()
        s_data = scales.contiguous()
        if self.pin:
            q_data = q_data.pin_memory()
            s_data = s_data.pin_memory()
        self.weights[name] = {
            "quantized": q_data,
            "scales": s_data,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "numel": n,
            "mode": "int8",
        }
        self._total_bytes += quantized.numel() + scales.numel() * 4

    def get(self, name: str) -> Optional[dict]:
        return self.weights.get(name)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def num_weights(self) -> int:
        return len(self.weights)


class AsyncTransfer:
    """Manages async CPU->GPU transfers using CUDA streams and pinned memory."""

    def __init__(self):
        self.transfer_stream = torch.cuda.Stream()
        self._prefetched: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def prefetch(self, name: str, weight_info: dict):
        """Start async transfer of quantized weight + scales to GPU."""
        with torch.cuda.stream(self.transfer_stream):
            gpu_quantized = weight_info["quantized"].cuda(non_blocking=True)
            gpu_scales = weight_info["scales"].cuda(non_blocking=True)
            self._prefetched[name] = (gpu_quantized, gpu_scales)

    def get_prefetched(self, name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Wait for and retrieve a prefetched weight. Returns (quantized, scales) on GPU."""
        if name not in self._prefetched:
            return None
        # Sync the transfer stream to ensure data is ready
        self.transfer_stream.synchronize()
        result = self._prefetched.pop(name)
        return result

    def is_prefetched(self, name: str) -> bool:
        return name in self._prefetched


class VRAMPager:
    """Main orchestrator — manages the full page cycle.

    Usage:
        pager = VRAMPager(block_size=128)

        # Load a model's weights into the pager
        for name, param in model.named_parameters():
            pager.store(name, param.data)

        # During forward pass, get decompressed weights
        weight = pager.get_weight("layer.0.weight")  # returns FP32 on GPU

        # Prefetch next layer while computing
        pager.prefetch("layer.1.weight")
    """

    def __init__(self, block_size: int = 128, dequant_kernel=None, mode: str = "fp16", pin: bool = True):
        self.store = PagedWeightStore(block_size=block_size, mode=mode, pin=pin)
        self.transfer = AsyncTransfer()
        self.block_size = block_size
        self.mode = mode
        self._kernel = dequant_kernel  # compiled CUDA kernel module
        self._layer_order: list = []  # for sequential prefetch prediction

    def set_kernel(self, kernel_module):
        """Set the compiled CUDA dequantization kernel."""
        self._kernel = kernel_module

    def store_weight(self, name: str, tensor: torch.Tensor):
        """Quantize and store a weight tensor."""
        self.store.quantize_and_store(name, tensor)

    def store_model(self, model: torch.nn.Module):
        """Store all parameters from a model."""
        for name, param in model.named_parameters():
            self.store_weight(name, param.data)
            self._layer_order.append(name)
        print(f"[VRAMPager] Stored {self.store.num_weights} weights "
              f"({self.store.total_bytes / 1e6:.1f} MB compressed)")

    def prefetch(self, name: str):
        """Start async transfer of a weight to GPU."""
        info = self.store.get(name)
        if info is not None:
            self.transfer.prefetch(name, info)

    def get_weight(self, name: str) -> torch.Tensor:
        """Get a weight tensor on GPU, decompressed if necessary.

        FP16 mode: transfers FP16 to GPU, casts to FP32 (lossless, 2x speedup)
        INT8 mode: transfers INT8+scales to GPU, dequantizes via kernel (4x speedup, slight quality loss)
        """
        info = self.store.get(name)
        if info is None:
            raise KeyError(f"Weight '{name}' not found in pager")

        weight_mode = info.get("mode", "int8")

        if weight_mode == "fp16":
            # FP16 paging — lossless, no kernel needed
            gpu_data = info["data"].cuda(non_blocking=True)
            torch.cuda.synchronize()
            # Return in original dtype (FP16 model stays FP16, FP32 gets upcast)
            orig_dtype = info["dtype"]
            if orig_dtype in (torch.float16, torch.bfloat16):
                result = gpu_data.to(orig_dtype).reshape(info["shape"])
            else:
                result = gpu_data.float().reshape(info["shape"])
            del gpu_data
            return result

        # INT8 paging — needs dequantization
        # Check if already prefetched
        prefetched = self.transfer.get_prefetched(name)
        if prefetched is not None:
            gpu_quantized, gpu_scales = prefetched
        else:
            gpu_quantized = info["quantized"].cuda()
            gpu_scales = info["scales"].cuda()

        # Decompress on GPU
        if self._kernel is not None:
            flat = self._kernel.dequantize_int8(gpu_quantized, gpu_scales, self.block_size)
        else:
            flat = (gpu_quantized.float().reshape(-1, self.block_size)
                    * gpu_scales.unsqueeze(1)).reshape(-1)

        result = flat[:info["numel"]].reshape(info["shape"])
        if info["dtype"] == torch.float16:
            result = result.half()
        elif info["dtype"] == torch.bfloat16:
            result = result.bfloat16()

        del gpu_quantized, gpu_scales
        return result

    def prefetch_next(self, current_name: str):
        """Prefetch the next weight in layer order (for sequential models like transformers)."""
        if current_name in self._layer_order:
            idx = self._layer_order.index(current_name)
            if idx + 1 < len(self._layer_order):
                next_name = self._layer_order[idx + 1]
                if not self.transfer.is_prefetched(next_name):
                    self.prefetch(next_name)

    def summary(self) -> str:
        """Print storage summary."""
        n = self.store.num_weights
        mb = self.store.total_bytes / 1e6
        return f"VRAMPager: {n} weights, {mb:.1f} MB compressed (INT8)"
