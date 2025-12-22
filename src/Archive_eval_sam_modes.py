# src/eval_sam_modes.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from segment_anything import SamPredictor
from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image
from src.metrics import compute_iou, compute_dice, compute_boundary_f1
from src.config import (
    RANKED_PROMPTS_DIR,
    VANILLA_PROMPTS_DIR,
)

def evaluate(
    dataset,
    mode="vanilla",          # "vanilla" | "ndvi" | "ndvi_lora"
    device="cuda",
    lora_ckpt="checkpoints/lora_mask_decoder.pt"
):
    assert mode in ["vanilla", "ndvi", "ndvi_lora"]

    # --------------------------------------------------
    # Load SAM
    # --------------------------------------------------
    sam, predictor, _ = load_sam_with_lora(device=device)

    if mode == "ndvi_lora":
        assert os.path.exists(lora_ckpt), "LoRA checkpoint not found"
        lora_state = torch.load(lora_ckpt, map_location=device)
        sam.load_state_dict(lora_state, strict=False)

    sam.eval()

    results = []

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    for item in dataset:
        image_id = item["id"]
        gt_mask_np = item["mask"]

        if gt_mask_np.sum() == 0:
            print(f"[SKIP] {image_id}: empty GT mask")
            continue

        # ---- Load image ----
        _, rgb = load_nc_image(item["nc_path"])
        predictor.set_image((rgb * 255).astype(np.uint8))

        # ---- Load prompt ----
        if mode == "vanilla":
            prompt_path = os.path.join(
                VANILLA_PROMPTS_DIR,
                f"vanilla_prompt_{image_id}.npy"
            )
        else:
            prompt_path = os.path.join(
                RANKED_PROMPTS_DIR,
                f"ranked_prompts_{image_id}.npy"
            )

        if not os.path.exists(prompt_path):
            print(f"[SKIP] {image_id}: prompt missing ({mode})")
            continue

        points = np.load(prompt_path)

        if len(points) == 0:
            print(f"[SKIP] {image_id}: empty prompt array")
            continue

        # ---- Predict mask ----
        with torch.no_grad():
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                multimask_output=False
            )

        pred_mask_np = masks[0].astype(np.uint8)

        # --------------------------------------------------
        # FIX 1: convert to torch tensors (CPU)
        # --------------------------------------------------
        pred_mask = torch.from_numpy(pred_mask_np).float()
        gt_mask = torch.from_numpy(gt_mask_np).float()

        # ---- Metrics ----
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        boundary = compute_boundary_f1(pred_mask, gt_mask)

        results.append((iou, dice, boundary))

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------
    if len(results) == 0:
        print(f"[{mode.upper()}] ❌ No valid samples evaluated.")
        return {"iou": np.nan, "dice": np.nan, "boundary": np.nan}

    results = np.array(results)

    metrics = {
        "iou": float(results[:, 0].mean()),
        "dice": float(results[:, 1].mean()),
        "boundary": float(results[:, 2].mean()),
    }

    print(f"[{mode.upper()}] Mean scores:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics
