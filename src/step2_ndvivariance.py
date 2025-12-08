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
    load_parcels,   # parcel filter
    load_mask,      # mask loader
)

# -------------------------------------------------------
# Extract numeric image ID from filename
# -------------------------------------------------------
def extract_image_id(nc_path):
    base = os.path.basename(nc_path)
    digits = ''.join([c for c in base if c.isdigit()])
    return int(digits)

# -------------------------------------------------------
# Load NDVI + RGB safely
# -------------------------------------------------------
def load_nc_image(path):

    ds = xr.open_dataset(path)
    print(f"Available variables in {os.path.basename(path)}: {list(ds.data_vars)}")
    best_t = 0

    # ---- NDVI ----
    ndvi = ds["NDVI"].isel(time=best_t).astype(np.float32).values
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    # ---- RGB ----
    R = ds["B4"].isel(time=best_t).astype(np.float32).values
    G = ds["B3"].isel(time=best_t).astype(np.float32).values
    B = ds["B2"].isel(time=best_t).astype(np.float32).values

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.nan_to_num(rgb)

    # Normalize RGB safely → 0..1 range
    min_val = rgb.min()
    rgb = rgb - min_val
    max_val = rgb.max()

    if max_val > 0:
        rgb = rgb / max_val
    else:
        rgb = np.zeros_like(rgb)

    rgb = np.clip(rgb, 0, 1)

    return ndvi, rgb


# -------------------------------------------------------
# Collect file paths
# -------------------------------------------------------
def collect_image_paths():
    nc_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nc")))
    mask_files = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))
    return nc_files, mask_files


# ------------------------------------------------------
# Compute NDVI temporal features
# ------------------------------------------------------

def compute_ndvi_temporal_features(nc_path):
    """
    Extracts ALL NDVI time steps from the .nc file and computes:
      • NDVI_mean
      • NDVI_variance
      • NDVI_max
      • NDVI_min

    Returns a dictionary of NDVI feature maps.
    """

    ds = xr.open_dataset(nc_path)

    if "NDVI" not in ds.data_vars:
        raise ValueError(f"NDVI not found in {nc_path}")

    # NDVI: shape [time, height, width]
    ndvi_ts = ds["NDVI"].astype(np.float32).values
    ndvi_ts = np.nan_to_num(ndvi_ts, nan=0.0)

    # ---- Compute temporal features ----
    ndvi_mean = np.mean(ndvi_ts, axis=0)
    ndvi_var  = np.var(ndvi_ts, axis=0)
    ndvi_max  = np.max(ndvi_ts, axis=0)
    ndvi_min  = np.min(ndvi_ts, axis=0)

    ds.close()
    return {
        "ndvi_mean": ndvi_mean,
        "ndvi_var": ndvi_var,
        "ndvi_max": ndvi_max,
        "ndvi_min": ndvi_min,
        "ndvi_ts": ndvi_ts,       # keep the full temporal cube (important later)
        "n_timesteps": ndvi_ts.shape[0]
    }

# -------------------------------------------------------
# Load all dataset components
# -------------------------------------------------------
def load_all_data():

    nc_files, mask_files = collect_image_paths()
    big_gdf = gpd.read_file(os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg"))

    dataset = []

    for idx, (nc_path, mask_path) in enumerate(zip(nc_files, mask_files)):

       # print(f"\nLoading image {idx+1} / {len(nc_files)}")
        #print(f" → File: {os.path.basename(nc_path)}")

        image_id = extract_image_id(nc_path)

        # Filter parcels
        parcels = load_parcels(big_gdf, image_id, country="Netherlands")
        #print(f" → Found {len(parcels)} parcels")

        # Load raster data
        ndvi_features = compute_ndvi_temporal_features(nc_path)
        

        # Load mask
        mask = load_mask(mask_path)

        dataset.append({
            "id": image_id,
            "ndvi_ts"   : ndvi_features["ndvi_ts"],
            "ndvi_mean" : ndvi_features["ndvi_mean"],
            "ndvi_var"  : ndvi_features["ndvi_var"],
            "ndvi_max"  : ndvi_features["ndvi_max"],
            "ndvi_min"  : ndvi_features["ndvi_min"],
            "n_timesteps": ndvi_features["n_timesteps"],
            "mask": mask,
            "parcels": parcels,
            "nc_path": nc_path,
            "mask_path": mask_path,
        })

        #print(" ✓ Step 2A Complete")

    return dataset

# -------------------------------------------------------
# Step 2B: NDVI Variance Mask
# -------------------------------------------------------

def compute_ndvi_variance_mask(ndvi_var, alpha=1.0):
    """
    A truly NDVI-guided threshold:
    threshold = mean + alpha · std
    """
    mu = ndvi_var.mean()
    sigma = ndvi_var.std()

    threshold = mu + alpha * sigma
    mask = ndvi_var > threshold

    return mask, threshold, mu, sigma
