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
from src.config import (
    TEMPORAL_PROMPTS_DIR,
    EVI2_PROMPTS_DIR,
    B2B3_PROMPTS_DIR,
    SPATIAL_PROMPTS_DIR,
    COMBINED_PROMPTS_DIR,
    ADAPTIVE_PROMPTS_DIR,
    VANILLA_PROMPTS_DIR,
)


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
    temporal_prompt_dir,
    prompt_type="temporal",  # "temporal" | "evi2" | "b2b3" | "spatial" | "combined" | "adaptive" | "vanilla"
    epochs=20,
    lr=1e-4,
    device="cuda",
    max_points=32,
    ckpt_path="checkpoints/lora_mask_decoder.pt",
    save_final=True,
    checkpoint_interval=5,
    resume_ckpt=None
):
    """
    Train LoRA adapters on SAM mask decoder only.
    Training-set metrics only (no validation).
    
    prompt_type: Type of prompts to use for training
        - "temporal": Temporal NDVI variance prompts
        - "evi2": EVI2 temporal enhancement prompts  
        - "b2b3": B2B3 gradient-based prompts
        - "spatial": Spatial gradient prompts
        - "combined": Combined temporal + spatial prompts
        - "adaptive": Adaptive prompts
        - "vanilla": Basic vanilla prompts
    
    save_final: Save model at training completion.
    checkpoint_interval: Save checkpoint every N epochs.
    resume_ckpt: Path to checkpoint to resume from.
    """

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # --------------------------------------------------
    # Prompt type mappings
    # --------------------------------------------------
    prompt_type_to_dir = {
        'temporal': temporal_prompt_dir,
        'evi2': EVI2_PROMPTS_DIR,
        'b2b3': B2B3_PROMPTS_DIR,
        'spatial': SPATIAL_PROMPTS_DIR,
        'combined': COMBINED_PROMPTS_DIR,
        'adaptive': ADAPTIVE_PROMPTS_DIR,
        'vanilla': VANILLA_PROMPTS_DIR,
    }
    
    prompt_type_to_filename = {
        'temporal': lambda image_id: f"superpixel_prompts_{image_id}.npy",
        'evi2': lambda image_id: f"evi2_prompts_{image_id}.npy",
        'b2b3': lambda image_id: f"b2b3_prompts_{image_id}.npy",
        'spatial': lambda image_id: f"spatial_prompts_{image_id}.npy",
        'combined': lambda image_id: f"combined_prompts_{image_id}.npy",
        'adaptive': lambda image_id: f"adaptive_prompts_{image_id}.npy",
        'vanilla': lambda image_id: f"vanilla_prompts_{image_id}.npy",
    }
    
    # Validate prompt_type
    if prompt_type not in prompt_type_to_dir:
        raise ValueError(f"Invalid prompt_type '{prompt_type}'. Valid options: {list(prompt_type_to_dir.keys())}")
    
    prompt_dir = prompt_type_to_dir[prompt_type]
    prompt_filename = prompt_type_to_filename[prompt_type]
    
    print(f"[INFO] Using {prompt_type} prompts from {prompt_dir}")

    # --------------------------------------------------
    # Load SAM + LoRA
    # --------------------------------------------------
    sam, predictor, lora_params = load_sam_with_lora(device=device)
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    print("[DEBUG] CUDA available:", torch.cuda.is_available())
    print("[DEBUG] SAM device:", next(sam.parameters()).device)
    print("[DEBUG] Trainable LoRA params:", sum(p.numel() for p in lora_params))

    # --------------------------------------------------
    # Resume from checkpoint if provided
    # --------------------------------------------------
    start_epoch = 0
    best_iou = -1.0
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location=device)
        sam.load_state_dict(checkpoint['model_state'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', -1.0)
        print(f"✅ Resumed from epoch {start_epoch}, best IoU {best_iou:.4f}")
    else:
        print("Starting training from scratch")

    sam.train()
    sam.mask_decoder.train()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(start_epoch, epochs):
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

            # ---- Load filtered prompts ----
            prompt_file = os.path.join(prompt_dir, prompt_filename(image_id))
            if not os.path.exists(prompt_file):
                print(f"  ⚠️  SKIP: {prompt_type} prompts not found for image {image_id}")
                continue
                
            points_np = np.load(prompt_file)[:max_points]

            points = torch.tensor(points_np, device=device, dtype=torch.float32)
            labels = torch.ones(len(points), device=device)

            # Pad to max_points if necessary
            n = len(points)
            if n < max_points:
                padding_points = torch.zeros(max_points - n, 2, device=device)
                padding_labels = torch.zeros(max_points - n, device=device)
                points = torch.cat([points, padding_points], dim=0)
                labels = torch.cat([labels, padding_labels], dim=0)

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
        # Save checkpoint every checkpoint_interval epochs
        # --------------------------------------------------
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_file = ckpt_path.replace('.pt', f'_ckpt_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
                'optimizer_state': optimizer.state_dict(),
                'best_iou': best_iou,
            }, ckpt_file)
            print(f"💾 Checkpoint saved at epoch {epoch+1}")

        # --------------------------------------------------
        # Save best model if IoU improves
        # --------------------------------------------------
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
                ckpt_path
            )
            print(f"✅ New best IoU {best_iou:.4f} — LoRA weights saved")

    # --------------------------------------------------
    # Save final model if requested
    # --------------------------------------------------
    if save_final:
        final_path = ckpt_path.replace('.pt', '_final.pt')
        torch.save(
            {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
            final_path
        )
        print(f"💾 Final model saved to {final_path}")

    print(f"\n🏁 Training finished | Best training IoU = {best_iou:.4f}")
    return sam
