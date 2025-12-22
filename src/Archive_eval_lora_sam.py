# src/eval_lora_sam.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image
from src.config import SAM_MODEL_TYPE, SAM_CHECKPOINT


# -------------------------
# Metrics
# -------------------------
def compute_iou(pred, gt):
    inter = ((pred > 0.5) & (gt > 0.5)).sum()
    union = ((pred > 0.5) | (gt > 0.5)).sum()
    return (inter / (union + 1e-6)).item()


def compute_dice(pred, gt):
    inter = (pred * gt).sum()
    return (2 * inter / (pred.sum() + gt.sum() + 1e-6)).item()


def boundary_f1(pred, gt):
    sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                         device=pred.device).float().view(1,1,3,3)

    def grad(x):
        gx = F.conv2d(x, sobel, padding=1)
        gy = F.conv2d(x, sobel.transpose(2,3), padding=1)
        return torch.sqrt(gx**2 + gy**2)

    pb = grad(pred.unsqueeze(0).unsqueeze(0))
    gb = grad(gt.unsqueeze(0).unsqueeze(0))

    return F.l1_loss(pb, gb).item()


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(
    dataset,
    ranked_prompt_dir,
    use_lora=False,
    device="cuda",
    ckpt_path="checkpoints/lora_mask_decoder.pt",
):
    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    if use_lora:
        sam, predictor, _ = load_sam_with_lora(device=device)
        lora_state = torch.load(ckpt_path, map_location=device)
        sam.load_state_dict(lora_state, strict=False)
        tag = "LoRA"
    else:
        # PURE BASELINE — NO LoRA
        from segment_anything import sam_model_registry
        from src.config import SAM_MODEL_TYPE, SAM_CHECKPOINT

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        predictor = SamPredictor(sam)
        tag = "Baseline"

    sam.eval()

    ious, dices, boundaries = [], [], []

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    for item in dataset:
        image_id = item["id"]
        gt_mask = item["mask"]

        if gt_mask.sum() == 0:
            continue

        _, rgb = load_nc_image(item["nc_path"])
        predictor.set_image((rgb * 255).astype(np.uint8))

        points_np = np.load(
            f"{ranked_prompt_dir}/ranked_prompts_{image_id}.npy"
        )

        points = torch.tensor(points_np, device=device).float()
        labels = torch.ones(len(points), device=device)

        masks, _, _ = predictor.predict_torch(
            point_coords=points.unsqueeze(0),
            point_labels=labels.unsqueeze(0),
            multimask_output=False,
        )

        pred = masks[0, 0]
        pred = (pred > 0.5).float()

        gt = torch.tensor(gt_mask, device=device).float()
        gt = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0),
            size=pred.shape,
            mode="nearest",
        )[0, 0]

        ious.append(compute_iou(pred, gt))
        dices.append(compute_dice(pred, gt))
        boundaries.append(boundary_f1(pred, gt))

    results = {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "boundary": float(np.mean(boundaries)),
    }

    print(f"\n[{tag}] Mean scores:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results