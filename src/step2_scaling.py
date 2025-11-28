import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd

from src.config import *
#from src.data_loader import load_nc_image, build_rgb_from_nc
#from src.ndvi_tools import select_best_ndvi_timestamp, compute_slic_superpixels
#from src.utils import ensure_dir

def extract_image_id(nc_path):
    base = os.path.basename(nc_path)
    digits = ''.join([c for c in base if c.isdigit()])
    return int(digits)


def load_nc_image(path):
    ds = xr.open_dataset(path)
    print(f"Available variables in {os.path.basename(path)}: {list(ds.data_vars)}")
    best_t = 0

    ndvi = ds["NDVI"].isel(time=best_t).astype(np.float32).values
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    R = ds["B4"].isel(time=best_t).astype(np.float32).values
    G = ds["B3"].isel(time=best_t).astype(np.float32).values
    B = ds["B2"].isel(time=best_t).astype(np.float32).values

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.nan_to_num(rgb)

    return ndvi, rgb

def load_mask(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def collect_image_paths():
    nc_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nc")))
    mask_files = sorted(glob.glob(os.path.join(MASKS_DIR, "*.tif")))

    return nc_files, mask_files


def load_all_data():
    nc_files,mask_files = collect_image_paths()

    # Load single parcel file gpkg
    big_gdf = gpd.read_file(os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg"))
    print(f"Loaded big Parcel file with {len(big_gdf)} features.")
    print(f"Columns: {list(big_gdf.columns)}")
    print(f"Data types:\n{big_gdf.dtypes}")
    print(f"'country' in columns: {'country' in big_gdf.columns}")
    print(f"'coutry' in columns: {'coutry' in big_gdf.columns}")

    dataset = []
    for idx, (nc_path, mask_path) in enumerate (zip(nc_files, mask_files)):
        print(f"Loading image {idx+1} / {len(nc_files)}")
        print(f" File: {os.path.basename(nc_path)}")

        #extract numeric image id
        image_id = extract_image_id(nc_path)

        #filter parcel for this image
        parcels = big_gdf[(big_gdf['coutry'] == "Netherlands") & (big_gdf['id'] == image_id)]
        print(f" Found {len(parcels)} parcels for this image.")

        #Load image + mask
        ndvi, rgb = load_nc_image(nc_path)
        mask = load_mask(mask_path)

        dataset.append({
            "id": image_id,
            "ndvi": ndvi,
            "rgb": rgb,
            "mask": mask,
            "parcels": parcels,
            "nc_path" : nc_path,
            "mask_path": mask_path,
        })

        print("\n Step 2A Complete: All images successfully loaded.")
    return dataset
