"""
PagedLinear — Drop-in replacement for torch.nn.Linear

Stores weights in compressed form (INT8/FP16) in CPU pinned memory.
On forward(), fetches compressed weights to GPU, decompresses via
CUDA kernel, applies any LoRA patches, runs matmul, and releases.

Compatible with ComfyUI's LoRA/ModelPatcher system — exposes a
.weight parameter so LoRA patches can find and modify it.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .memory_manager import VRAMPager


class PagedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with compressed VRAM paging.

    Key design: keeps a small meta-tensor as self.weight so that
    model.state_dict() includes it and ComfyUI's LoRA system can
    find the key and register patches. The actual data lives in
    the VRAMPager. During forward(), we fetch the paged weight,
    apply any accumulated LoRA patches, then compute.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 pager: Optional[VRAMPager] = None, weight_name: str = ""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pager = pager
        self.weight_name = weight_name

        # Store a CPU FP16 copy of the weight as the parameter.
        # This is what ComfyUI's LoRA system sees and patches.
        # It lives in CPU RAM (not VRAM) and is small (FP16 = half the FP32 size).
        # The pager's compressed version is used for fast GPU transfer during forward().
        self.weight = None  # Set by from_linear()

        # Bias stays on GPU (tiny, not worth paging)
        self.bias: Optional[nn.Parameter] = None

        # When ComfyUI patches weight via LoRA, we store only the DELTA
        # (patched_weight - original_weight) in CPU RAM. This is small
        # because LoRA changes are low-rank. Applied during forward().
        self._patched_weight_delta: Optional[torch.Tensor] = None

        # LoRA patches accumulated by ComfyUI's ModelPatcher
        # Format: list of (strength, patch_data, strength_model, offset, function)
        self._lora_patches: List[Tuple] = []

    def set_weight(self, tensor: torch.Tensor, **kwargs):
        """Called by ComfyUI's ModelPatcher when applying LoRA patches.

        ComfyUI's patch_weight_to_device() flow:
        1. get_key_weight() returns self.weight (our meta tensor)
        2. It fetches the real weight via convert_weight(), applies LoRA
        3. It calls set_weight(patched_weight) to write it back

        We compute the delta (patched - original) and store ONLY the delta.
        This keeps memory low: LoRA deltas are small (rank * dims * 4 bytes).
        During forward(), we decompress + add delta in one shot.
        """
        # Get the original unpatched weight to compute delta
        original = self.pager.get_weight(self.weight_name)
        delta = (tensor - original).to("cpu")  # Store delta in CPU RAM, not VRAM
        self._patched_weight_delta = delta
        del original, tensor

    def convert_weight(self, tensor: torch.Tensor, **kwargs):
        """Called by ComfyUI to convert weight format.

        Returns the actual decompressed weight from the pager so ComfyUI
        can apply LoRA patches to real data instead of the meta tensor.
        The result gets passed to set_weight() after patching.
        """
        return self.pager.get_weight(self.weight_name)

    @classmethod
    def from_linear(cls, linear: nn.Linear, pager: VRAMPager, weight_name: str = "") -> "PagedLinear":
        """Create a PagedLinear from an existing nn.Linear, storing weights in the pager."""
        paged = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            pager=pager,
            weight_name=weight_name,
        )

        # Store weight in pager (compressed in pinned CPU memory)
        pager.store_weight(weight_name, linear.weight.data)

        # Keep a CPU FP16 copy as the nn.Parameter for LoRA compatibility.
        # ComfyUI's ModelPatcher will see this, patch it with LoRA deltas,
        # and write it back. During forward(), if LoRA was applied (weight
        # is on a real device), we use the patched weight. Otherwise we
        # use the pager's compressed version.
        paged.weight = nn.Parameter(
            linear.weight.data.half().cpu(),
            requires_grad=False
        )

        # Keep bias on GPU (it's tiny)
        if linear.bias is not None:
            paged.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        return paged

    def _apply_lora_patches(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply accumulated LoRA patches to the decompressed weight.

        This mimics what ComfyUI's calculate_weight() does, but applied
        to our paged weight after decompression.
        """
        if not self._lora_patches:
            return weight

        for patch_info in self._lora_patches:
            strength, patch_data, strength_model, offset, function = patch_info
            try:
                # Standard LoRA patch: weight += strength * (alpha/dim) * (up @ down)
                if isinstance(patch_data, tuple) and len(patch_data) >= 2:
                    # LoRA format: (up, down, alpha, ...)
                    up = patch_data[0].to(weight.device, dtype=weight.dtype)
                    down = patch_data[1].to(weight.device, dtype=weight.dtype)
                    alpha = patch_data[2] if len(patch_data) > 2 else None

                    if alpha is not None:
                        dim = down.shape[0]
                        scale = alpha / dim
                    else:
                        scale = 1.0

                    weight = weight + (strength * scale * strength_model) * (up @ down)
                elif callable(function):
                    weight = function(weight, patch_data, strength)
            except Exception as e:
                # Don't crash on patch failure — log and continue
                print(f"[VRAMPager] Warning: LoRA patch failed for {self.weight_name}: {e}")

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prefetch next layer while we compute this one
        if self.pager is not None:
            self.pager.prefetch_next(self.weight_name)

        if self.weight is not None and self.weight.device.type != "cpu":
            # Weight was moved to GPU by ComfyUI's patch system (LoRA applied)
            # Use it directly — it has LoRA deltas baked in
            weight = self.weight.float()
        else:
            # No LoRA — use the pager's compressed path (fast)
            weight = self.pager.get_weight(self.weight_name)

            # Apply LoRA delta if set via set_weight()
            if self._patched_weight_delta is not None:
                weight = weight + self._patched_weight_delta.to(weight.device, dtype=weight.dtype)

            # Apply any manually added LoRA patches
            if self._lora_patches:
                weight = self._apply_lora_patches(weight)

        # Standard linear: y = x @ W^T + b
        output = torch.nn.functional.linear(x, weight, self.bias)

        # Release the decompressed weight from GPU
        del weight

        return output

    def extra_repr(self) -> str:
        lora_str = f", loras={len(self._lora_patches)}" if self._lora_patches else ""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, paged=True{lora_str}")


def replace_linear_with_paged(model: nn.Module, pager: VRAMPager,
                                prefix: str = "") -> nn.Module:
    """Replace all nn.Linear layers in a model with PagedLinear.

    Walks the module tree, replaces each Linear with a PagedLinear
    that stores its weights in the VRAMPager. The replacement maintains
    a meta-tensor .weight parameter so that LoRA patching systems
    (like ComfyUI's ModelPatcher) can find and register patches.

    Args:
        model: The model to modify (modified in-place)
        pager: VRAMPager instance to store weights
        prefix: Current module path prefix (for recursion)

    Returns:
        The modified model
    """
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            weight_name = f"{full_name}.weight"
            paged = PagedLinear.from_linear(child, pager, weight_name=weight_name)
            setattr(model, name, paged)
        else:
            # Recurse into child modules
            replace_linear_with_paged(child, pager, prefix=full_name)

    return model
