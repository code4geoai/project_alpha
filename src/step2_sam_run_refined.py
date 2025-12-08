# src/step2_sam_run_refined.py

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config import (
    REFINED_DIR, PRED_MASK_DIR, PRED_VIS_DIR,
    SUMMARY_CSV
)

from src.sam_tools import load_sam_model, calculate_iou,binarize_mask
from src.step2_scaling import load_nc_image


# --------------------------------------------------
# Helper for SAM prediction
# --------------------------------------------------
def predict_from_points(predictor, rgb_image, points_xy):
    if points_xy is None or len(points_xy) == 0:
        return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    point_coords = points_xy.astype(np.float32)
    point_labels = np.ones(len(point_coords), dtype=np.int32)

    # SAM expects uint8 image set beforehand
    if rgb_image.max() <= 1.1:
        predictor.set_image((rgb_image * 255).astype(np.uint8))
    else:
        predictor.set_image(rgb_image.astype(np.uint8))

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    return masks, scores


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def run_sam_with_refined_prompts(dataset):
    predictor = load_sam_model()
    rows = []

    print("\n========== Step 2D: Running SAM with Refined Prompts ==========\n")

    for item in dataset:
        image_id = item["id"]
        #rgb = item.get("rgb")
        ndvi_tmp, rgb = load_nc_image(item["nc_path"])
        gt_mask = item.get("mask")

        print(f"\n[RUN-REFINED] Image {image_id}")

        # Safety check — skip if GT mask is empty
        if gt_mask is None or gt_mask.sum() == 0:
            print(f"  [SKIP] Image {image_id}: GT empty — not evaluated.")
            continue

        h, w = gt_mask.shape

        # Load refined prompts
        rp_path = os.path.join(REFINED_DIR, f"refined_prompts_{image_id}.npy")
        if not os.path.exists(rp_path):
            print(f"  [SKIP] Missing refined prompts ({rp_path})")
            continue

        centroids = np.load(rp_path)

        # Ensure SAM (x,y) coordinate order
        r0, c0 = centroids[0]
        if r0 <= h and c0 <= w:
            points_xy = np.vstack([centroids[:, 1], centroids[:, 0]]).T
        else:
            points_xy = centroids.copy()

        # SAM limit
        MAX_POINTS = 200
        if len(points_xy) > MAX_POINTS:
            points_xy = points_xy[:MAX_POINTS]

        # Run SAM
        try:
            masks, scores = predict_from_points(predictor, rgb, points_xy)
        except Exception as e:
            print("  [ERROR] SAM predictor failed:", e)
            continue

        # Pick best mask by IoU
        best_iou = -1
        best_mask = None
        best_score = None

        for k, m in enumerate(masks):
            pred = binarize_mask(m)
            iou = calculate_iou(pred, gt_mask)

            if iou > best_iou:
                best_iou = iou
                best_mask = pred
                best_score = float(scores[k])

        # Save mask
        os.makedirs(PRED_MASK_DIR, exist_ok=True)
        np.save(os.path.join(PRED_MASK_DIR, f"pred_mask_refined_{image_id}.npy"),
                best_mask.astype(np.uint8))

        # Save visualization
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(np.clip(rgb, 0, 1)); ax[0].set_title("RGB"); ax[0].axis("off")
        ax[1].imshow(gt_mask, cmap="gray"); ax[1].set_title("GT Mask"); ax[1].axis("off")
        ax[2].imshow(best_mask, cmap="gray"); ax[2].set_title(f"SAM (IoU={best_iou:.3f})"); ax[2].axis("off")

        fig.savefig(os.path.join(PRED_VIS_DIR, f"vis_refined_{image_id}.png"), dpi=150)
        plt.close(fig)

        # Compute metrics
        gt_vec = (gt_mask > 0).astype(int).ravel()
        pred_vec = best_mask.astype(int).ravel()

        prec = precision_score(gt_vec, pred_vec, zero_division=0)
        rec = recall_score(gt_vec, pred_vec, zero_division=0)
        f1 = f1_score(gt_vec, pred_vec, zero_division=0)

        print(f"  -> Refined prompts={len(points_xy)}, IoU={best_iou:.3f}, F1={f1:.3f}")

        rows.append({
            "image_id": image_id,
            "n_prompts_refined": len(points_xy),
            "best_iou": float(best_iou),
            "pred_score": best_score,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

    # Save summary CSV
    df = pd.DataFrame(rows).sort_values("image_id")
    df.to_csv(SUMMARY_CSV.replace(".csv", "_refined.csv"), index=False)

    print("\nSaved summary:", SUMMARY_CSV.replace(".csv", "_refined.csv"))
    print(df)
