# src/eval_gold_standard.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image
from src.metrics import compute_iou, compute_dice, compute_boundary_f1
from src.config import (
    SAM_MODEL_TYPE,
    SAM_CHECKPOINT,
    TEMPORAL_PROMPTS_DIR,
    VANILLA_PROMPTS_DIR,
)


@torch.no_grad()
def evaluate(
    dataset,
    mode="vanilla",      # "vanilla" | "ndvi" | "ndvi_lora"
    device="cuda",
    lora_ckpt="checkpoints/lora_mask_decoder.pt",
):
    """
    Gold-standard evaluator for SAM ablations.
    """

    assert mode in ["vanilla", "ndvi", "ndvi_lora"]

    # --------------------------------------------------
    # Load SAM
    # --------------------------------------------------
    if mode == "vanilla":
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        predictor = SamPredictor(sam)

    else:
        sam, predictor, _ = load_sam_with_lora(device=device)
        if mode == "ndvi_lora":
            assert os.path.exists(lora_ckpt), "LoRA checkpoint missing"
            lora_state = torch.load(lora_ckpt, map_location=device)
            sam.load_state_dict(lora_state, strict=False)

    sam.eval()

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    ious, dices, boundaries = [], [], []

    for item in dataset:
        image_id = item["id"]
        gt_mask_np = item["mask"]

        if gt_mask_np.sum() == 0:
            continue

        # ---- Load image ----
        _, rgb = load_nc_image(item["nc_path"])
        predictor.set_image((rgb * 255).astype(np.uint8))

        # ---- Load prompts ----
        if mode == "vanilla":
            prompt_path = os.path.join(
                VANILLA_PROMPTS_DIR,
                f"vanilla_prompts_{image_id}.npy"
            )
        else:
            prompt_path = os.path.join(
                TEMPORAL_PROMPTS_DIR,
                f"superpixel_prompts_{image_id}.npy"
            )

        if not os.path.exists(prompt_path):
            continue

        points_np = np.load(prompt_path)
        if len(points_np) == 0:
            continue

        points = torch.tensor(points_np, device=device).float()
        labels = torch.ones(len(points), device=device)

        # ---- Predict (TORCH ONLY) ----
        masks, _, _ = predictor.predict_torch(
            point_coords=points.unsqueeze(0),
            point_labels=labels.unsqueeze(0),
            multimask_output=False,
        )

        pred = (masks[0, 0] > 0.5).float()

        gt = torch.tensor(gt_mask_np, device=device).float()
        gt = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0),
            size=pred.shape,
            mode="nearest",
        )[0, 0]

        # ---- Metrics ----
        ious.append(compute_iou(pred, gt))
        dices.append(compute_dice(pred, gt))
        boundaries.append(compute_boundary_f1(pred, gt))

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------
    results = {
        "iou": float(np.mean(ious)) if len(ious) else np.nan,
        "dice": float(np.mean(dices)) if len(dices) else np.nan,
        "boundary": float(np.mean(boundaries)) if len(boundaries) else np.nan,
    }

    print(f"\n[{mode.upper()}] Mean scores:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results
