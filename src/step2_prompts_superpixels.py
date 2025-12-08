import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

from src.config import PROMPTS_DIR

os.makedirs(PROMPTS_DIR, exist_ok=True)


def generate_superpixel_prompts(ndvi_var, ndvi_mask, image_id, n_segments=150):
    """
    Generate NDVI-guided superpixel centroids.
    Only pixels where ndvi_mask == 1 are used.
    """

    # -------------------------
    # 1. Mask-out low-variance pixels
    # -------------------------
    masked_ndvi = np.zeros_like(ndvi_var)
    masked_ndvi[ndvi_mask == 1] = ndvi_var[ndvi_mask == 1]

    if masked_ndvi.sum() == 0:
        print(f"[WARN] Image {image_id}: No active NDVI variance pixels!")
        return []

    # -------------------------
    # 2. Run SLIC only on nonzero NDVI variance areas
    # -------------------------
    ndvi_norm = (masked_ndvi - masked_ndvi.min()) / (masked_ndvi.max() - masked_ndvi.min() + 1e-6)

    segments = slic(
        ndvi_norm,
        n_segments=n_segments,
        compactness=0.1,
        start_label=1,
        mask=ndvi_mask.astype(bool),
        channel_axis=None
    )

    # -------------------------
    # 3. Extract centroids from superpixels
    # -------------------------
    props = regionprops(segments)

    centroids = []
    for p in props:
        r, c = p.centroid
        centroids.append((float(c), float(r)))   # (x, y)

    centroids = np.array(centroids)

    # -------------------------
    # 4. Save prompts
    # -------------------------
    out_path = os.path.join(PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy")
    np.save(out_path, centroids)

    print(f"[2D.A] Image {image_id}: {len(centroids)} superpixel prompts saved → {out_path}")

    return centroids


def run_superpixel_generation(dataset):
    """
    dataset = output of Step 2A (NDVI temporal features)
    For each image, run NDVI-guided superpixel prompt extraction.
    """

    results = {}

    for item in dataset:
        img_id = item["id"]

        ndvi_var = item["ndvi_var"]
        mask = item["ndvi_mask"]     # this must come from Step 2B

        print(f"\n[2D.A] Processing image {img_id}")

        centroids = generate_superpixel_prompts(
            ndvi_var=ndvi_var,
            ndvi_mask=mask,
            image_id=img_id
        )

        results[img_id] = {
            "centroids": centroids,
            "n_centroids": len(centroids)
        }

    return results
