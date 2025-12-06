import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import geopandas as gpd
import xarray as xr
import rasterio
import numpy as np
from src.config import *


def load_nc_image(path):
    ds = xr.open_dataset(path)
    return ds

def build_rgb_from_nc(ds, t=0):
    B2 = ds["B2"].isel(time=t).values.astype(np.float32)
    B3 = ds["B3"].isel(time=t).values.astype(np.float32)
    B4 = ds["B4"].isel(time=t).values.astype(np.float32)

    rgb = np.dstack((B4, B3, B2))
    rgb_min, rgb_max = np.percentile(rgb, (2, 98))
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb_norm = np.clip(rgb_norm, 0, 1)

    return rgb_norm

def load_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
        #print(src.crs)
    return mask



def load_parcels(big_gdf, image_id, country="Netherlands"):
    """
    Filter parcels from the big GPKG file based on:
    - country name
    - id == image_id (numeric portion of file name)
    """
    subset = big_gdf[
        (big_gdf["coutry"] == country) &
        (big_gdf["id"] == image_id)
    ]

    return subset

