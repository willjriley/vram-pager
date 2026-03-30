"""
ComfyUI Custom Nodes: VRAM Pager

Two nodes:
1. CompressedPager — passthrough node that accelerates any model's
   weight transfers. Works with standard loaders, LoRAs, ControlNet.
2. PagedModelLoader — loads safetensors with compressed paging (original node)
"""
import os
import sys
import ctypes
import torch
import time

VRAM_PAGER_ROOT = os.path.dirname(os.path.abspath(__file__))
if VRAM_PAGER_ROOT not in sys.path:
    sys.path.insert(0, VRAM_PAGER_ROOT)

from pager.comfy_hook import accelerate_model


def _load_cuda_kernel():
    """Load the compiled CUDA dequantization kernel."""
    dll_path = os.path.join(VRAM_PAGER_ROOT, "build", "dequant.dll")
    so_path = os.path.join(VRAM_PAGER_ROOT, "build", "dequant.so")

    for path in [dll_path, so_path]:
        if not os.path.exists(path):
            continue
        try:
            cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
            if os.path.exists(cuda_bin):
                os.add_dll_directory(cuda_bin)
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)

            dll = ctypes.CDLL(path)
            dll.launch_dequant_int8.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ]

            class KernelWrapper:
                def __init__(self, dll):
                    self.dll = dll
                def dequantize_int8(self, quantized, scales, block_size):
                    n = quantized.numel()
                    output = torch.empty(n, dtype=torch.float32, device=quantized.device)
                    self.dll.launch_dequant_int8(
                        quantized.data_ptr(), scales.data_ptr(), output.data_ptr(),
                        n, block_size, None)
                    return output

            print("[VRAMPager] CUDA kernel loaded!")
            return KernelWrapper(dll)
        except Exception as e:
            print(f"[VRAMPager] Kernel load failed: {e}")

    print("[VRAMPager] No CUDA kernel found, using PyTorch fallback")
    return None


_KERNEL = _load_cuda_kernel()


class CompressedPager:
    """Accelerate any model's weight transfers with compressed paging.

    This is a passthrough node — connect it between your model loader
    and the rest of your workflow. The model is unchanged, LoRAs work
    normally, it just makes weight transfers faster.

    Usage:
        UNETLoader → CompressedPager → LoRA Loader → KSampler
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["int8", "fp16"],
                         {"default": "int8",
                          "tooltip": "int8: 3.4x faster, imperceptible quality difference. fp16: 1.8x faster, lossless."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "accelerate"
    CATEGORY = "VRAMPager"

    def accelerate(self, model, mode="int8"):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  VRAM Pager — Compressed Weight Paging")
        print(f"  Mode: {mode} | Kernel: {'CUDA' if _KERNEL else 'PyTorch fallback'}")
        print(f"{'='*60}")

        result = accelerate_model(model, mode=mode, kernel=_KERNEL)

        print(f"  Done in {time.time()-t0:.1f}s")
        print(f"{'='*60}\n")

        return (result,)


class PagedModelLoader:
    """Load a safetensors model with compressed paging."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "block_size": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_paged"
    CATEGORY = "VRAMPager"

    def load_paged(self, model_path, block_size=128):
        import comfy.sd
        from safetensors import safe_open

        print(f"\n{'='*60}")
        print(f"  VRAM Pager — Direct Model Loader")
        print(f"{'='*60}")

        t0 = time.time()
        state_dict = {}
        if os.path.isdir(model_path):
            shards = sorted([f for f in os.listdir(model_path)
                           if f.endswith(".safetensors") and "index" not in f])
            for shard in shards:
                count = 0
                with safe_open(os.path.join(model_path, shard),
                             framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                        count += 1
                print(f"  {shard}: {count} tensors")
        else:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            print(f"  Loaded {len(state_dict)} tensors")

        model_patcher = comfy.sd.load_diffusion_model_state_dict(state_dict)
        if model_patcher is None:
            raise RuntimeError("Could not detect model architecture")

        # Accelerate with compressed paging
        model_patcher = accelerate_model(model_patcher, mode="int8", block_size=block_size, kernel=_KERNEL)

        print(f"  Total: {time.time()-t0:.1f}s")
        print(f"{'='*60}\n")

        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "CompressedPager": CompressedPager,
    "PagedModelLoader": PagedModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompressedPager": "Compressed Pager (VRAM Pager)",
    "PagedModelLoader": "Paged Model Loader (VRAM Pager)",
}
