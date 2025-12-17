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


def generate_superpixel_prompts(
        ndvi_var,
        ndvi_mask,
        image_id,
        n_segments=150,
        min_region_size=20
):
    """
    NDVI-variance-guided superpixel prompt generation.
    """

    # -------------------------
    # 1. Restrict analysis only to active NDVI variance areas
    # -------------------------
    masked_ndvi = np.zeros_like(ndvi_var)
    masked_ndvi[ndvi_mask == 1] = ndvi_var[ndvi_mask == 1]

    if masked_ndvi.sum() == 0:
        print(f"[WARN] Image {image_id}: No active NDVI variance pixels → Skipping.")
        return np.array([])

    # -------------------------
    # 2. Safe normalization
    # -------------------------
    vmin = masked_ndvi.min()
    vmax = masked_ndvi.max()

    if vmax - vmin < 1e-6:
        print(f"[WARN] Image {image_id}: NDVI variance too flat → Skipping.")
        return np.array([])

    ndvi_norm = (masked_ndvi - vmin) / (vmax - vmin)

    # -------------------------
    # 3. SLIC Superpixels *restricted by mask*
    # -------------------------
    segments = slic(
        ndvi_norm,
        n_segments=n_segments,
        compactness=0.1,
        start_label=1,
        mask=ndvi_mask.astype(bool),
        channel_axis=None
    )

    # Clean small objects (noise superpixels after masking)
    segments = remove_small_objects(segments, min_region_size)

    # -------------------------
    # 4. Extract centroids
    # -------------------------
    props = regionprops(segments)

    if len(props) < 3:
        print(f"[WARN] Image {image_id}: Too few superpixels → {len(props)}")
    
    centroids = []
    for p in props:
        r, c = p.centroid
        centroids.append((float(c), float(r)))   # x, y (col, row)

    centroids = np.array(centroids)

    # -------------------------
    # 5. Save
    # -------------------------
    out_path = os.path.join(PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy")
    np.save(out_path, centroids)

    print(f"[2D] Image {image_id}: {len(centroids)} superpixel prompts saved → {out_path}")

    return centroids


def run_superpixel_generation(dataset):
    """
    dataset = output of Step 2A (with ndvi_var)
    PLUS Step 2B (with ndvi_mask)
    """

    results = {}

    for item in dataset:
        img_id = item["id"]

        ndvi_var = item["ndvi_var"]
        ndvi_mask = item["ndvi_mask"]

        print(f"\n[2D] Generating superpixel prompts for image {img_id}...")

        centroids = generate_superpixel_prompts(
            ndvi_var=ndvi_var,
            ndvi_mask=ndvi_mask,
            image_id=img_id
        )

        results[img_id] = {
            "centroids": centroids,
            "n_centroids": len(centroids)
        }

    print("\n[2D] Superpixel prompt generation complete for all images.")

    return results
