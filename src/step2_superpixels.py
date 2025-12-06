import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
import json
import os

def generate_superpixel_centroids(ndvi, num_segments=300, compactness=10):
    """
    Generates superpixels and extracts centroid points.
    Returns: list of [x, y] coordinates usable by SAM.
    """

    # Run SLIC superpixels
    segments = slic(
        ndvi,
        n_segments=num_segments,
        compactness=compactness,
        sigma=1,
        start_label=1,
        channel_axis=None,

    )

    props = regionprops(segments)

    centroids = []
    for region in props:
        y, x = region.centroid
        centroids.append([int(x), int(y)])

    return centroids, segments


def save_centroids(image_id, centroids, out_dir="results/prompts"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{image_id}_centroids.json")

    with open(out_path, "w") as f:
        json.dump({"image_id": image_id, "centroids": centroids}, f, indent=2)

    print(f"[2D.1] Saved centroids for {image_id}: {len(centroids)} points")
    return out_path


def run_superpixel_generation_for_all(images_dict, out_dir="results/prompts"):
    """
    images_dict = {
        image_id: {"ndvi": ndvi_array, "rgb": rgb_array, "mask": mask_array}
    }
    """

    for image_id, data in images_dict.items():

        ndvi = data["ndvi"]

        print(f"\n[2D.1] Processing {image_id} ...")

        centroids, segments = generate_superpixel_centroids(ndvi)

        save_centroids(image_id, centroids, out_dir=out_dir)

    print("\n=== Step 2D.1 Completed: Superpixel prompts generated ===")
