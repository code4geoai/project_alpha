# src/generate_vanilla_prompts.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import geopandas as gpd
import xarray as xr

from src.config import (
    IMAGES_DIR,
    VECTOR_DIR,
    VANILLA_PROMPTS_DIR,
)
from src.step2_scaling import collect_image_paths, extract_image_id
from src.data_loader import load_parcels


def map_to_pixel(x_coord, y_coord, x_vals, y_vals):
    """
    Convert map coordinates → pixel indices
    """
    x_pixel = int(np.argmin(np.abs(x_vals - x_coord)))
    y_pixel = int(np.argmin(np.abs(y_vals - y_coord)))
    return x_pixel, y_pixel


def generate_vanilla_centroid_prompts(dataset=None, country="Netherlands"):
    """
    Generate ONE vanilla prompt per parcel:
    - centroid of each parcel
    - converted correctly to pixel coordinates
    - saved as .npy
    If dataset is provided, limit to those images.
    """

    os.makedirs(VANILLA_PROMPTS_DIR, exist_ok=True)

    print("📥 Loading parcel vector file...")
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )
    big_gdf = big_gdf[big_gdf["coutry"] == country]
    big_gdf["parcel_id"] = range(len(big_gdf))

    nc_files, _ = collect_image_paths()

    if dataset is not None:
        image_ids = [item["id"] for item in dataset]
        nc_files = [p for p in nc_files if extract_image_id(p) in image_ids]

    print("🚀 Generating vanilla centroid prompts...\n")

    for nc_path in nc_files:
        image_id = extract_image_id(nc_path)

        parcels = load_parcels(big_gdf, image_id, country)
        if len(parcels) == 0:
            print(f"SKIP: No parcels for image {image_id}")
            continue

        # --------------------------------------------------
        # Load NetCDF spatial axes
        # --------------------------------------------------
        ds = xr.open_dataset(nc_path)
        x_vals = ds["x"].values
        y_vals = ds["y"].values

        centroids = []
        for idx, parcel in parcels.iterrows():
            # --------------------------------------------------
            # Compute centroid
            # --------------------------------------------------
            centroid_geom = parcel.geometry.centroid

            # --------------------------------------------------
            # Convert centroid → pixel coordinates (CORRECT)
            # --------------------------------------------------
            x_pixel, y_pixel = map_to_pixel(
                centroid_geom.x, centroid_geom.y, x_vals, y_vals
            )

            # --------------------------------------------------
            # Skip if centroid maps to outer borders
            # --------------------------------------------------
            if (x_pixel == 0 or x_pixel == len(x_vals) - 1 or
                y_pixel == 0 or y_pixel == len(y_vals) - 1):
                #print(f"SKIP: Parcel {parcel['parcel_id']} (Image {image_id}) centroid on border")
                continue

            centroids.append([x_pixel, y_pixel])

            print(
                f"✔ Parcel {parcel['parcel_id']} (Image {image_id}) | "
                f"Centroid map=({centroid_geom.x:.1f}, {centroid_geom.y:.1f}) "
                f"→ pixel=({x_pixel}, {y_pixel})"
            )

        # Save all centroids for the image
        if centroids:
            out_path = os.path.join(
                VANILLA_PROMPTS_DIR,
                f"vanilla_prompts_{image_id}.npy"
            )
            np.save(out_path, np.array(centroids, dtype=np.float32))
            print(f"[Vanilla] Saved {len(centroids)} centroids → {out_path}")

    print("\n✅ Vanilla prompt generation completed.")



    
