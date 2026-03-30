"""
Model Converter — Quantize and store model weights for paged inference.

Takes a safetensors model file, quantizes all weights to INT8,
and saves in a format optimized for the VRAMPager.
"""
import torch
import os
import json
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict


def quantize_state_dict(state_dict: Dict[str, torch.Tensor],
                        block_size: int = 128) -> Dict[str, dict]:
    """Quantize all tensors in a state dict to INT8 with per-block scales.

    Returns a dict of {name: {"quantized": INT8, "scales": FP32, "shape": tuple}}
    """
    result = {}
    total_original = 0
    total_compressed = 0

    for name, tensor in state_dict.items():
        flat = tensor.float().flatten()
        n = flat.numel()
        total_original += n * 4  # FP32 = 4 bytes

        # Pad to block boundary
        padded_n = ((n + block_size - 1) // block_size) * block_size
        if padded_n > n:
            flat = torch.cat([flat, torch.zeros(padded_n - n)])

        # Per-block quantization
        blocked = flat.reshape(-1, block_size)
        scales = blocked.abs().max(dim=1).values / 127.0
        scales = scales.clamp(min=1e-10)
        quantized = (blocked / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        result[name] = {
            "quantized": quantized.flatten()[:n].contiguous(),
            "scales": scales.contiguous(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": n,
        }
        total_compressed += n + scales.numel() * 4  # INT8 + FP32 scales

    ratio = total_original / total_compressed if total_compressed > 0 else 0
    print(f"Quantized {len(result)} tensors: "
          f"{total_original/1e9:.2f} GB -> {total_compressed/1e9:.2f} GB "
          f"({ratio:.1f}x compression)")

    return result


def save_paged_model(quantized: Dict[str, dict], output_dir: str):
    """Save quantized model in paged format.

    Creates:
      output_dir/weights.int8.safetensors  (quantized weights)
      output_dir/scales.fp32.safetensors   (scale factors)
      output_dir/metadata.json              (shapes, dtypes, config)
    """
    os.makedirs(output_dir, exist_ok=True)

    weights = {}
    scales = {}
    metadata = {}

    for name, info in quantized.items():
        weights[name] = info["quantized"]
        scales[f"{name}.scales"] = info["scales"]
        metadata[name] = {
            "shape": info["shape"],
            "dtype": info["dtype"],
            "numel": info["numel"],
        }

    save_file(weights, os.path.join(output_dir, "weights.int8.safetensors"))
    save_file(scales, os.path.join(output_dir, "scales.fp32.safetensors"))

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "format": "vram_pager_v1",
            "quantization": "int8_per_block",
            "block_size": 128,
            "num_tensors": len(metadata),
            "tensors": metadata,
        }, f, indent=2)

    print(f"Saved paged model to {output_dir}/")


def load_paged_model(model_dir: str) -> Dict[str, dict]:
    """Load a paged model back into memory."""
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)

    result = {}
    with safe_open(os.path.join(model_dir, "weights.int8.safetensors"),
                   framework="pt", device="cpu") as wf:
        with safe_open(os.path.join(model_dir, "scales.fp32.safetensors"),
                       framework="pt", device="cpu") as sf:
            for name, info in meta["tensors"].items():
                result[name] = {
                    "quantized": wf.get_tensor(name),
                    "scales": sf.get_tensor(f"{name}.scales"),
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "numel": info["numel"],
                }

    return result


def convert_safetensors(input_path: str, output_dir: str, block_size: int = 128):
    """Convert a safetensors model file to paged format.

    Args:
        input_path: Path to .safetensors file (or directory with sharded files)
        output_dir: Where to save the paged model
        block_size: Quantization block size (default 128)
    """
    print(f"Loading {input_path}...")

    state_dict = {}
    if os.path.isdir(input_path):
        # Sharded model
        for f in sorted(os.listdir(input_path)):
            if f.endswith(".safetensors") and "index" not in f:
                print(f"  Loading shard: {f}")
                with safe_open(os.path.join(input_path, f),
                               framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        state_dict[key] = sf.get_tensor(key)
    else:
        # Single file
        with safe_open(input_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")

    quantized = quantize_state_dict(state_dict, block_size=block_size)
    save_paged_model(quantized, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert.py <input_safetensors> <output_dir>")
        print("Example: python convert.py models/wan2.1-14b.safetensors models/wan2.1-14b-paged/")
        sys.exit(1)

    convert_safetensors(sys.argv[1], sys.argv[2])
