import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import rasterio
import geopandas as gpd
import shapely
from rasterio.plot import show
from shapely.geometry import mapping


def rasterize_parcels(parcels, shape):
    if len(parcels) == 0:
        return np.zeros(shape, dtype=np.uint8)

    geoms = [(geom, 1) for geom in parcels.geometry]
    parcels_mask = rasterio.features.rasterize(
        geoms,
        out_shape=shape,
        fill=0,
        dtype=np.uint8
    )
    return parcels_mask


def visualize_alignment(image_id, ndvi, rgb, mask, parcels, out_dir="results/visual_checks"):
    """
    Creates a 5-panel visualization:
    1. RGB
    2. NDVI
    3. Mask
    4. RGB + parcel polygons
    5. Mask + parcel polygons
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    # -------------------------
    # 1. RGB
    # -------------------------
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title(f"RGB – Image {image_id}")
    axes[0].axis("off")

    # -------------------------
    # 2. NDVI
    # -------------------------
    axes[1].imshow(ndvi, cmap="RdYlGn")
    axes[1].set_title("NDVI")
    axes[1].axis("off")

    # -------------------------
    # 3. Mask
    # -------------------------
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    # rasterize parcels into mask
    parcel_mask = rasterize_parcels(parcels, ndvi.shape)

    # Panel 4: RGB + parcels
    axes[3].imshow(rgb)
    axes[3].imshow(parcel_mask, cmap="autumn", alpha=0.4)
    axes[3].set_title("RGB + Parcel Boundaries")
    axes[3].axis("off")

    # Panel 5: Mask + parcels
    axes[4].imshow(mask, cmap="gray")
    axes[4].imshow(parcel_mask, cmap="autumn", alpha=0.4)
    axes[4].set_title("Mask + Parcel Boundaries")
    axes[4].axis("off")

    # Save
    out_path = os.path.join(out_dir, f"ALIGN_{image_id}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved alignment visualization: {out_path}")
