# src/lora_train.py
# LoRA training for SAM — TRAINING ONLY (no validation)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import torch.nn.functional as F

from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image


# -------------------------
# Metric helpers
# -------------------------
def compute_iou(pred, gt, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)


# -------------------------
# Training
# -------------------------
def train_lora(
    dataset,
    ranked_prompt_dir,
    epochs=20,
    lr=1e-4,
    device="cuda",
    max_points=32,
    ckpt_path="checkpoints/lora_mask_decoder.pt"
):
    """
    Train LoRA adapters on SAM mask decoder only.
    Training-set metrics only (no validation).
    """

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # --------------------------------------------------
    # Load SAM + LoRA
    # --------------------------------------------------
    sam, predictor, lora_params = load_sam_with_lora(device=device)
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    print("[DEBUG] CUDA available:", torch.cuda.is_available())
    print("[DEBUG] SAM device:", next(sam.parameters()).device)
    print("[DEBUG] Trainable LoRA params:", sum(p.numel() for p in lora_params))

    best_iou = -1.0

    sam.train()
    sam.mask_decoder.train()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(epochs):
        print(f"\n[LoRA] Epoch {epoch+1}/{epochs}")

        epoch_loss = 0.0
        epoch_iou = 0.0
        n_samples = 0

        for i, item in enumerate(dataset):
            image_id = item["id"]
            gt_mask = item["mask"]

            if gt_mask.sum() == 0:
                continue

            print(f"  → Image {image_id} ({i+1}/{len(dataset)})")

            # ---- Load RGB ----
            _, rgb = load_nc_image(item["nc_path"])
            predictor.set_image((rgb * 255).astype(np.uint8))

            # ---- Load ranked prompts ----
            points_np = np.load(
                os.path.join(ranked_prompt_dir, f"ranked_prompts_{image_id}.npy")
            )[:max_points]

            points = torch.tensor(points_np, device=device, dtype=torch.float32)
            labels = torch.ones(len(points), device=device)

            # ---- Image embeddings (cached) ----
            image_embeddings = predictor.get_image_embedding()

            # ---- Prompt encoder (frozen) ----
            with torch.no_grad():
                sparse_pe, dense_pe = sam.prompt_encoder(
                    points=(points.unsqueeze(0), labels.unsqueeze(0)),
                    boxes=None,
                    masks=None,
                )
                image_pe = sam.prompt_encoder.get_dense_pe()

            # ---- Mask decoder (LoRA trainable) ----
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_pe,
                dense_prompt_embeddings=dense_pe,
                multimask_output=False,
            )

            pred_logits = low_res_logits[0, 0]
            pred_prob = torch.sigmoid(pred_logits)

            gt = torch.tensor(gt_mask, device=device).float()
            gt_low = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0),
                size=pred_logits.shape[-2:],
                mode="nearest",
            )[0, 0]

            # ---- Loss ----
            loss = F.binary_cross_entropy_with_logits(pred_logits, gt_low)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Metrics ----
            iou = compute_iou(pred_prob, gt_low)

            epoch_loss += loss.item()
            epoch_iou += iou.item()
            n_samples += 1

        mean_loss = epoch_loss / max(n_samples, 1)
        mean_iou = epoch_iou / max(n_samples, 1)

        print(
            f"[Epoch {epoch+1}] "
            f"Mean loss = {mean_loss:.4f} | Mean IoU = {mean_iou:.4f}"
        )

        # --------------------------------------------------
        # Save ONLY if IoU improves
        # --------------------------------------------------
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
                ckpt_path
            )
            print(f"✅ New best IoU {best_iou:.4f} — LoRA weights saved")

    print(f"\n🏁 Training finished | Best training IoU = {best_iou:.4f}")
    return sam
