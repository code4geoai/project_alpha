# src/step0_ndvi_timeseries.py
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import xarray as xr
from tqdm import tqdm
from src.config import DATA_DIR

def compute_ndvi_and_evi2_time_series(nc_path):
    """
    Reads B4,B8 for ALL timesteps in the .nc file and returns:
    - ndvi_stack: (T, H, W) float32
    - ndvi_var:   (H, W) temporal variance
    - evi2_stack: (T, H, W) float32
    - evi2_var:   (H, W) temporal variance
    """
    ds = xr.open_dataset(nc_path)

    if "B4" not in ds or "B8" not in ds:
        raise ValueError(f"Dataset missing B4/B8 bands: {nc_path}")

    B4 = ds["B4"].values.astype(np.float32)   # shape (T,H,W)
    B8 = ds["B8"].values.astype(np.float32)

    # NDVI computation
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi_var = np.var(ndvi, axis=0).astype(np.float32)

    # EVI2 computation
    evi2 = 2.5 * (B8 - B4) / (B8 + 4.4 * B4 + 1)
    evi2 = np.clip(evi2, -1, 1)  # Optional clipping for consistency
    evi2_var = np.var(evi2, axis=0).astype(np.float32)

    return ndvi, ndvi_var, evi2, evi2_var


def attach_ndvi_and_evi2_timeseries_to_dataset(dataset):
    """
    For each item, compute NDVI[T] and EVI2[T] stacks and variances ONCE
    and attach them into the dataset dict.
    """

    print("\n===== Step-0: Computing NDVI and EVI2 Time-Series for All Images =====\n")

    for item in tqdm(dataset):
        nc_path = item["nc_path"]
        ndvi_stack, ndvi_var, evi2_stack, evi2_var = compute_ndvi_and_evi2_time_series(nc_path)

        item["ndvi_stack"] = ndvi_stack     # (T,H,W)
        item["ndvi_var"] = ndvi_var         # (H,W)
        item["evi2_stack"] = evi2_stack     # (T,H,W)
        item["evi2_var"] = evi2_var         # (H,W)

    print("\n>>> Step-0 COMPLETED: NDVI and EVI2 time-series added to dataset.\n")

    return dataset


# Backward compatibility: keep original functions
def compute_ndvi_time_series(nc_path):
    """
    Legacy function: Computes only NDVI time series.
    """
    ndvi, ndvi_var, _, _ = compute_ndvi_and_evi2_time_series(nc_path)
    return ndvi, ndvi_var


def attach_ndvi_timeseries_to_dataset(dataset):
    """
    Legacy function: Attaches only NDVI time series.
    """
    print("\n===== Step-0: Computing NDVI Time-Series for All Images (Legacy) =====\n")

    for item in tqdm(dataset):
        nc_path = item["nc_path"]
        ndvi_stack, ndvi_var = compute_ndvi_time_series(nc_path)

        item["ndvi_stack"] = ndvi_stack     # (T,H,W)
        item["ndvi_var"] = ndvi_var         # (H,W)

    print("\n>>> Step-0 COMPLETED: NDVI time-series added to dataset.\n")

    return dataset
