# src/lora_train.py
# Step L1 — Minimal, correct LoRA training loop for SAM

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import torch.nn.functional as F

from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image


def train_lora(
    dataset,
    ranked_prompt_dir,
    epochs=1,
    lr=1e-4,
    device="cuda",
    max_points=32          # 🔴 VERY IMPORTANT
):
    # --------------------------------------------------
    # Load SAM + LoRA
    # --------------------------------------------------
    sam, predictor, lora_params = load_sam_with_lora(device=device)

    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # ---- sanity checks ----
    print("[DEBUG] CUDA available:", torch.cuda.is_available())
    print("[DEBUG] SAM device:", next(sam.parameters()).device)
    print("[DEBUG] Trainable LoRA params:", sum(p.numel() for p in lora_params))

    sam.train()
    sam.mask_decoder.train()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(epochs):
        print(f"\n[LoRA] Epoch {epoch+1}/{epochs}")

        for i, item in enumerate(dataset):
            image_id = item["id"]
            gt_mask = item["mask"]

            if gt_mask.sum() == 0:
                continue

            print(f"  → Image {image_id} ({i+1}/{len(dataset)})")

            # ---- Load RGB (CPU → SAM handles GPU internally) ----
            _, rgb = load_nc_image(item["nc_path"])
            predictor.set_image((rgb * 255).astype(np.uint8))

            # ---- Load ranked prompts ----
            prompt_path = os.path.join(
                ranked_prompt_dir,
                f"ranked_prompts_{image_id}.npy"
            )

            points_np = np.load(prompt_path)

            # 🔴 LIMIT PROMPTS DURING TRAINING
            if len(points_np) > max_points:
                points_np = points_np[:max_points]

            points = torch.tensor(
                points_np,
                dtype=torch.float32,
                device=device
            )

            labels = torch.ones(len(points), device=device)

            # ---- SAM forward (MASK DECODER ONLY is trainable) ----
            # `SamPredictor.predict_torch()` is wrapped in torch.no_grad() in the upstream
            # Segment Anything code, which prevents gradients.
            # For LoRA training, bypass predictor.predict_torch and call the SAM modules
            # directly, keeping encoders frozen and enabling grads only for mask_decoder.

            # Image embeddings are cached inside predictor after `set_image`.
            # They can be treated as constants (no grad needed) since we only train mask_decoder.
            image_embeddings = predictor.get_image_embedding()

            # Prompt encoder is frozen; compute embeddings without grad.
            with torch.no_grad():
                sparse_prompt_embeddings, dense_prompt_embeddings = sam.prompt_encoder(
                    points=(points.unsqueeze(0), labels.unsqueeze(0)),
                    boxes=None,
                    masks=None,
                )
                image_pe = sam.prompt_encoder.get_dense_pe()

            # Mask decoder contains LoRA trainable params; run with grad enabled.
            low_res_logits, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
            )

            pred_logits = low_res_logits[0, 0]  # [h, w] logits

            gt = torch.tensor(
                gt_mask,
                dtype=torch.float32,
                device=device
            )

            # Downsample GT to match SAM low-res logits resolution.
            gt_low = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0),
                size=pred_logits.shape[-2:],
                mode="nearest",
            )[0, 0]

            # ---- Loss ----
            # Use logits-safe BCE for stability and correct gradients.
            loss = F.binary_cross_entropy_with_logits(pred_logits, gt_low)

            # Optional boundary term on probabilities (same resolution as logits).
            pred_prob = torch.sigmoid(pred_logits)
            loss = loss + 0.3 * boundary_loss(pred_prob, gt_low)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            print(f"     loss = {loss.item():.4f}")

        print(f"✔ Epoch {epoch+1} completed")

    return sam
def boundary_loss(pred, gt):
    """
    Simple boundary emphasis using Sobel gradients
    """
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                           device=pred.device).float().view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    pred_gx = F.conv2d(pred.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
    pred_gy = F.conv2d(pred.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)

    gt_gx = F.conv2d(gt.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
    gt_gy = F.conv2d(gt.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)

    return F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)

