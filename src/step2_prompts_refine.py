import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.config import *

os.makedirs(REFINED_DIR, exist_ok=True)

def refine_prompts_by_ndvi_variance(dataset, top_percent=0.25):
    """
    Refine superpixel centroids by keeping only those inside the top-X% NDVI variance pixels.
    dataset: list of dictionaries built in previous steps
             each item must contain:
             - "id"
             - "ndvi_var": (H,W) ndarray
             - "ndvi_mask": (H,W) ndarray (from Step 2B)
    top_percent: fraction to keep (0.25 = keep top 25% NDVI variance)
    """

    print("\n========== Step 2B-REFINE: Reducing prompts per image using NDVI variance ==========\n")
    results = []

    for item in dataset:
        img_id = item["id"]
        ndvi_var = item.get("ndvi_var")
        ndvi_mask = item.get("ndvi_mask")

        # If GT is empty, skip image (1023, 1100)
        if item.get("mask") is None or item["mask"].sum() == 0:
            print(f"[SKIP] Image {img_id}: GT empty — no refinement needed.")
            continue

        print(f"[REFINE] Image {img_id}")

        # Path to original superpixel centroids
        original_file = os.path.join(PROMPTS_DIR, f"superpixel_prompts_{img_id}.npy")
        if not os.path.exists(original_file):
            print(f"  -> No superpixel prompts found. Skipping.")
            continue

        centroids = np.load(original_file)
        H, W = ndvi_var.shape

        # ---- Step 1: Compute NDVI variance threshold ----
        flat = ndvi_var.flatten()
        cutoff = np.quantile(flat[flat > 0], 1 - top_percent)
        high_mask = (ndvi_var >= cutoff).astype(np.uint8)

        active_pixels = high_mask.sum()
        print(f"  NDVI cutoff={cutoff:.5f}, active_pixels={active_pixels}")

        # ---- Step 2: Filter centroids that fall inside the high-var mask ----
        refined = []
        dropped = 0

        for (row, col) in centroids:
            r, c = int(row), int(col)
            if 0 <= r < H and 0 <= c < W:
                if high_mask[r, c] == 1:
                    refined.append([c, r])  # always save as (x,y)
                else:
                    dropped += 1

        refined = np.array(refined, dtype=np.float32)

        # ---- Save ----
        out_path = os.path.join(REFINED_DIR, f"refined_prompts_{img_id}.npy")
        np.save(out_path, refined)

        print(f"  -> Original={len(centroids)}, Refined={len(refined)}, Dropped={dropped}")
        results.append((img_id, len(centroids), len(refined)))

    print("\nSaved refined prompts to:", REFINED_DIR)
    return results