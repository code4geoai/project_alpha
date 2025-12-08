# src/step0_ndvi_timeseries.py
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import xarray as xr
from tqdm import tqdm
from src.config import DATA_DIR

def compute_ndvi_time_series(nc_path):
    """
    Reads B4,B8 for ALL timesteps in the .nc file and returns:
    - ndvi_stack: (T, H, W) float32
    - ndvi_var:   (H, W) temporal variance
    """
    ds = xr.open_dataset(nc_path)

    if "B4" not in ds or "B8" not in ds:
        raise ValueError(f"Dataset missing B4/B8 bands: {nc_path}")

    B4 = ds["B4"].values.astype(np.float32)   # shape (T,H,W)
    B8 = ds["B8"].values.astype(np.float32)

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)       # avoid division zero
    ndvi = np.clip(ndvi, -1, 1)

    ndvi_var = np.var(ndvi, axis=0).astype(np.float32)

    return ndvi, ndvi_var


def attach_ndvi_timeseries_to_dataset(dataset):
    """
    For each item, compute NDVI[T] stack and variance ONCE
    and attach them into the dataset dict.
    """

    print("\n===== Step-0: Computing NDVI Time-Series for All Images =====\n")

    for item in tqdm(dataset):
        nc_path = item["nc_path"]
        ndvi_stack, ndvi_var = compute_ndvi_time_series(nc_path)

        item["ndvi_stack"] = ndvi_stack     # (T,H,W)
        item["ndvi_var"] = ndvi_var         # (H,W)

    print("\n>>> Step-0 COMPLETED: NDVI time-series added to dataset.\n")

    return dataset
