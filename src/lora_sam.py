# src/lora_sam.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from src.config import SAM_CHECKPOINT, SAM_MODEL_TYPE


# -------------------------
# LoRA Linear (SAFE)
# -------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r

        # Create LoRA params on the SAME device/dtype as the wrapped Linear.
        # This avoids CPU/CUDA mismatches when injecting after `sam.to(device)`.
        w = self.linear.weight
        self.lora_A = nn.Parameter(torch.zeros((r, linear.in_features), device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros((linear.out_features, r), device=w.device, dtype=w.dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # freeze base
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        # Works for both [B, C] and [B, N, C] (matmul broadcasts over leading dims)
        lora = (x.matmul(self.lora_A.t()).matmul(self.lora_B.t())) * self.scaling
        return base + lora


# -------------------------
# Inject LoRA safely
# -------------------------
def inject_lora_into_mask_decoder(mask_decoder, r=8, alpha=16):
    # Collect first, then replace, to avoid mutating the module tree while iterating.
    target_names = []
    for name, module in mask_decoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
            target_names.append(name)

    for name in target_names:
        parent = mask_decoder
        *path, last = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        module = getattr(parent, last)
        if isinstance(module, LoRALinear):
            continue
        setattr(parent, last, LoRALinear(module, r=r, alpha=alpha))


def freeze_all_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


# -------------------------
# Loader
# -------------------------
def load_sam_with_lora(device="cuda", r=8, alpha=16):
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)

    # Freeze everything first; only LoRA params should train.
    freeze_all_params(sam)

    # Inject LoRA into mask decoder Linear projections.
    inject_lora_into_mask_decoder(sam.mask_decoder, r=r, alpha=alpha)

    # Move the whole model (including newly-created LoRA params) to target device.
    sam.to(device)

    lora_params = [p for p in sam.parameters() if p.requires_grad]

    print(f"[LoRA] Trainable params: {sum(p.numel() for p in lora_params):,}")

    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    return sam, predictor, lora_params
