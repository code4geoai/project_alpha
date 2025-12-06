import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd

from src.config import *
from src.data_loader import (
    load_parcels,          # parcel filter
    load_mask,             # mask loader
    build_rgb_from_nc      # RGB builder
)

# -------------------------------------------------------
# Extract numeric image ID from filename
# -------------------------------------------------------
def extract_image_id(nc_path):
    base = os.path.basename(nc_path)
    digits = ''.join([c for c in base if c.isdigit()])
    return int(digits)

# -------------------------------------------------------
# Load NDVI + RGB from .nc file
# -------------------------------------------------------
def load_nc_image(path):
    
    ds = xr.open_dataset(path)
    print(f"Available variables in {os.path.basename(path)}: {list(ds.data_vars)}")
    best_t = 0

    # Load NDVI
    ndvi = ds["NDVI"].isel(time=best_t).astype(np.float32).values
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Load RGB
    R = ds["B4"].isel(time=best_t).astype(np.float32).values
    G = ds["B3"].isel(time=best_t).astype(np.float32).values
    B = ds["B2"].isel(time=best_t).astype(np.float32).values

    rgb = np.stack([R, G, B], axis=-1)

    # Normalize safely
    rgb = rgb - np.nanmin(rgb)
    max_val = np.nanmax(rgb)
    if max_val > 0:
        rgb = rgb / max_val
    else:
        rgb = np.zeros_like(rgb)

    rgb = np.clip(rgb, 0, 1)
    return ndvi, rgb


# -------------------------------------------------------
# Collect file paths for all images + masks
# -------------------------------------------------------
def collect_image_paths():
    nc_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nc")))
    mask_files = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))
    return nc_files, mask_files

# -------------------------------------------------------
# Load all dataset components for 10 images
# -------------------------------------------------------
def load_all_data():

    nc_files, mask_files = collect_image_paths()

    # Load BIG parcel gpkg only once
    big_gdf = gpd.read_file(os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg"))

    dataset = []

    for idx, (nc_path, mask_path) in enumerate(zip(nc_files, mask_files)):

        print(f"\nLoading image {idx+1} / {len(nc_files)}")
        print(f" → File: {os.path.basename(nc_path)}")

        # numeric image id
        image_id = extract_image_id(nc_path)

        # filter parcels for this image
        parcels = load_parcels(big_gdf, image_id, country="Netherlands")
        print(f" → Found {len(parcels)} parcels")

        # load NDVI + RGB
        ndvi, rgb = load_nc_image(nc_path)

        # load mask
        mask = load_mask(mask_path)

        # append structured dataset
        dataset.append({
            "id": image_id,
            "ndvi": ndvi,
            "rgb": rgb,
            "mask": mask,
            "parcels": parcels,
            "nc_path": nc_path,
            "mask_path": mask_path,
        })

        print(" ✓ Step 2A Complete")

    return dataset
