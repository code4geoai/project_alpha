import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from skimage.measure import regionprops

from src.config import TEMPORAL_PROMPTS_DIR, RANKED_PROMPTS_DIR

os.makedirs(RANKED_PROMPTS_DIR, exist_ok=True)

"""
Ranks temporal superpixel prompts using temporal saliency.
Keeps top-K centroids per image.
"""


# -------------------------------------------------------
# Rank centroids by saliency strength
# -------------------------------------------------------
def rank_temporal_prompts(
    labels,
    centroids,
    saliency,
    top_k=100
):
    """
    labels   : [H,W] superpixel labels
    centroids: list[(x,y)] of centroids to rank (assumed filtered)
    saliency : [H,W] temporal saliency map

    Returns: ranked_centroids (top_k from centroids)
    """

    props = regionprops(labels, intensity_image=saliency)

    scores = []
    for r in props:
        score = r.mean_intensity
        y, x = r.centroid
        centroid = (float(x), float(y))

        # Check if this centroid is in centroids (with tolerance)
        is_in = any(
            np.allclose(centroid, c, atol=1e-2) for c in centroids
        )
        if is_in:
            scores.append((score, float(x), float(y)))

    # Sort descending by saliency strength
    scores.sort(key=lambda x: x[0], reverse=True)

    top = scores[:top_k]
    ranked_centroids = [(x, y) for (_, x, y) in top]

    return np.array(ranked_centroids, dtype=np.float32)


# -------------------------------------------------------
# Save ranked prompts
# -------------------------------------------------------
def save_ranked_prompts(image_id, centroids):
    out_path = os.path.join(
        RANKED_PROMPTS_DIR,
        f"ranked_prompts_{image_id}.npy"
    )
    np.save(out_path, centroids)
    print(f"[RANK] Image {image_id}: saved {len(centroids)} prompts → {out_path}")
    return out_path
