# ============================================================
# Step-0 + Step-2A : Temporal NDVI Extraction & Statistics
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import numpy as np
import xarray as xr
import geopandas as gpd

from src.config import *
from src.data_loader import load_parcels, load_mask


# -------------------------------------------------------
# Extract numeric image ID from filename
# -------------------------------------------------------
def extract_image_id(nc_path):
    base = os.path.basename(nc_path)
    digits = ''.join([c for c in base if c.isdigit()])
    return int(digits)

# ----------------------------------------------------------
# Load RGB
# ----------------------------------------------------------
def build_rgb_from_nc(ds, t=0):
    B2 = ds["B2"].isel(time=t).values.astype(np.float32)
    B3 = ds["B3"].isel(time=t).values.astype(np.float32)
    B4 = ds["B4"].isel(time=t).values.astype(np.float32)

    rgb = np.dstack((B4, B3, B2))
    rgb_min, rgb_max = np.percentile(rgb, (2, 98))
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb_norm = np.clip(rgb_norm, 0, 1)

    return rgb_norm


# -------------------------------------------------------
# Step-0 : Load TRUE NDVI and EVI2 time series
# -------------------------------------------------------
def compute_ndvi_and_evi2_temporal_features(nc_path):
    """
    Load ALL NDVI[t] and compute EVI2[t] frames and compute:
        • ndvi_ts        (T,H,W)
        • evi2_ts        (T,H,W)
        • ndvi_mean, evi2_mean
        • ndvi_var, evi2_var
        • ndvi_min/max, evi2_min/max
    """

    ds = xr.open_dataset(nc_path)

    if "B4" not in ds or "B8" not in ds:
        raise ValueError(f"B4/B8 bands not found in {nc_path}")

    B4 = ds["B4"].values.astype(np.float32)
    B8 = ds["B8"].values.astype(np.float32)

    # Compute NDVI
    ndvi_ts = (B8 - B4) / (B8 + B4 + 1e-6)
    ndvi_ts = np.clip(ndvi_ts, -1, 1)

    # Compute EVI2
    evi2_ts = 2.5 * (B8 - B4) / (B8 + 4.4 * B4 + 1)
    evi2_ts = np.clip(evi2_ts, -1, 1)

    ndvi_mean = ndvi_ts.mean(axis=0)
    ndvi_var  = ndvi_ts.var(axis=0)
    ndvi_max  = ndvi_ts.max(axis=0)
    ndvi_min  = ndvi_ts.min(axis=0)

    evi2_mean = evi2_ts.mean(axis=0)
    evi2_var  = evi2_ts.var(axis=0)
    evi2_max  = evi2_ts.max(axis=0)
    evi2_min  = evi2_ts.min(axis=0)

    ds.close()

    return {
        "ndvi_ts": ndvi_ts,
        "evi2_ts": evi2_ts,
        "ndvi_mean": ndvi_mean,
        "ndvi_var": ndvi_var,
        "ndvi_max": ndvi_max,
        "ndvi_min": ndvi_min,
        "evi2_mean": evi2_mean,
        "evi2_var": evi2_var,
        "evi2_max": evi2_max,
        "evi2_min": evi2_min,
        "n_timesteps": ndvi_ts.shape[0]
    }


# -------------------------------------------------------
# Step-2B : NDVI variance → mask
# -------------------------------------------------------
def compute_ndvi_variance_mask(ndvi_var, alpha=1.0):
    """
    threshold = mean + alpha * std
    """
    mu = ndvi_var.mean()
    sigma = ndvi_var.std()

    threshold = mu + alpha * sigma
    mask = ndvi_var > threshold

    return mask.astype(np.uint8), threshold, mu, sigma


# -------------------------------------------------------
# Collect image + mask paths
# -------------------------------------------------------
def collect_image_paths():
    nc_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nc")))
    mask_files = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))
    return nc_files, mask_files


# -------------------------------------------------------
# Step-0 + Step-2A : Build dataset dictionary
# -------------------------------------------------------
def load_all_data():

    nc_files, mask_files = collect_image_paths()
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )

    os.makedirs(os.path.join(RESULTS_DIR, "ndvi_masks"), exist_ok=True)

    dataset = []

    for nc_path, mask_path in zip(nc_files, mask_files):

        image_id = extract_image_id(nc_path)

        # ---- Load parcels
        parcels = load_parcels(big_gdf, image_id, country="Netherlands")

        # ---- NDVI and EVI2 temporal features
        features = compute_ndvi_and_evi2_temporal_features(nc_path)

        # ---- Load RGB (t=0) ----
        ds = xr.open_dataset(nc_path)
        rgb = build_rgb_from_nc(ds, t=0)
        ds.close()


        # ---- NDVI variance → binary mask
        ndvi_mask, thr, mu, sigma = compute_ndvi_variance_mask(
            features["ndvi_var"], alpha=1.0
        )

        # Save mask for debugging
        np.save(
            os.path.join(RESULTS_DIR, "ndvi_masks", f"ndvi_mask_{image_id}.npy"),
            ndvi_mask
        )

        # ---- Load GT mask (parcel segmentation)
        gt_mask = load_mask(mask_path)

        dataset.append({
            "id": image_id,
            "rgb": rgb,
            "ndvi_ts": features["ndvi_ts"],  # Keep original name for compatibility
            "ndvi_stack": features["ndvi_ts"],  # Also add stack for consistency
            "evi2_stack": features["evi2_ts"],  # Added EVI2 stack
            "ndvi_mean": features["ndvi_mean"],
            "ndvi_var": features["ndvi_var"],
            "ndvi_max": features["ndvi_max"],
            "ndvi_min": features["ndvi_min"],
            "evi2_mean": features["evi2_mean"],
            "evi2_var": features["evi2_var"],
            "evi2_max": features["evi2_max"],
            "evi2_min": features["evi2_min"],
            "n_timesteps": features["n_timesteps"],
            "ndvi_mask": ndvi_mask,          # <-- IMPORTANT for Superpixels
            "mask": gt_mask,
            "parcels": parcels,
            "nc_path": nc_path,
            "mask_path": mask_path
        })

        print(
            f"[Step-0] Image {image_id}: NDVI and EVI2 time-series loaded "
            f"(T={features['n_timesteps']}) | "
            f"NDVI variance threshold={thr:.4f}"
        )

    return dataset
