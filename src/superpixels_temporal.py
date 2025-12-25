import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.util import img_as_float


from src.config import TEMPORAL_PROMPTS_DIR, VECTOR_DIR

import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from src.step2_scaling import collect_image_paths, extract_image_id
from src.data_loader import load_parcels

os.makedirs(TEMPORAL_PROMPTS_DIR, exist_ok=True)

"""
superpixels_temporal.py

Uses ONLY:
    ✔ NDVI temporal variance map
    ✔ NDVI peak-difference map

to generate superpixel prompt centroids.
"""


# -------------------------------------------------------
# NDVI TEMPORAL VARIANCE
# -------------------------------------------------------
def compute_ndvi_variance(ndvi_ts):
    """
    ndvi_ts: NDVI time-series [T, H, W]
    returns: variance map [H, W]
    """
    return np.var(ndvi_ts.astype(np.float32), axis=0)


# -------------------------------------------------------
# NDVI PEAK DIFFERENCE
# -------------------------------------------------------
def compute_ndvi_peak_diff(ndvi_ts):
    """
    ndvi_ts: NDVI time-series [T, H, W]
    returns: peak difference map = max(ndvi) - min(ndvi)
    """
    ndvi_max = ndvi_ts.max(axis=0)
    ndvi_min = ndvi_ts.min(axis=0)
    return (ndvi_max - ndvi_min).astype(np.float32)


# -------------------------------------------------------
# COMBINED SALIENCY MASK
# Option 2 = NDVI_var_mask + peak_diff_mask
# -------------------------------------------------------
def build_temporal_saliency_mask(ndvi_ts, var_thresh=0.05, peak_thresh=0.15):
    """
    Combine NDVI variance + peak-diff into pixel saliency mask.
    Returns float32 mask [H,W] in 0..1
    """

    # 1. Compute features
    var_map = compute_ndvi_variance(ndvi_ts)
    peak_map = compute_ndvi_peak_diff(ndvi_ts)

    # Normalize to [0..1]
    def norm(x):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    var_n = norm(var_map)
    peak_n = norm(peak_map)

    # 2. Threshold masks
    var_mask = (var_n > var_thresh).astype(np.float32)
    peak_mask = (peak_n > peak_thresh).astype(np.float32)

    # 3. Combine (OR-like)
    combined = np.clip(var_mask + peak_mask, 0, 1)

    return combined.astype(np.float32)


# -------------------------------------------------------
# SUPERPIXELS FROM TEMPORAL SALIENCY
# -------------------------------------------------------
def generate_temporal_superpixels(
    rgb,
    ndvi_ts,
    n_segments=1200,
    compactness=0.1,
    var_thresh=0.05,
    peak_thresh=0.15
):
    """
    rgb: float32 [H,W,3] in 0..1
    ndvi_ts: NDVI time-series [T,H,W]

    Returns:
        labels: SLIC labels [H,W] int
        centroids: list of (x, y)
    """

    # ---- Build temporal saliency channel ----
    saliency = build_temporal_saliency_mask(
        ndvi_ts,
        var_thresh=var_thresh,
        peak_thresh=peak_thresh
    )

    # Add saliency as 4th channel
    rgb_f = img_as_float(rgb)
    stacked = np.dstack([rgb_f, saliency])

    # ---- Run SLIC ----
    labels = slic(
        stacked,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        channel_axis=-1
    ).astype(np.int32)

    # ---- Extract centroids ----
    regions = regionprops(labels)

    centroids = []
    for r in regions:
        y, x = r.centroid
        centroids.append((float(x), float(y)))

    return labels, centroids, saliency


# -------------------------------------------------------
# FILTER CENTROIDS WITHIN PARCELS
# -------------------------------------------------------
def filter_centroids_within_parcels(centroids, parcels, x_vals, y_vals):
    """
    centroids: list of (x_pixel, y_pixel)
    parcels: GeoDataFrame of parcels
    x_vals, y_vals: 1D arrays from NetCDF

    Returns: filtered list of (x_pixel, y_pixel) within any parcel
    """
    filtered = []
    for x_pixel, y_pixel in centroids:
        # Convert pixel to map coordinates
        x_map = x_vals[int(x_pixel)]
        y_map = y_vals[int(y_pixel)]
        point = Point(x_map, y_map)

        # Check if point is within any parcel
        within = False
        for _, parcel in parcels.iterrows():
            if parcel.geometry.contains(point):
                within = True
                break
        if within:
            filtered.append((x_pixel, y_pixel))

    return filtered


# -------------------------------------------------------
# SELECT TOP SALIENCY PIXELS PER PARCEL
# -------------------------------------------------------
def select_top_saliency_pixels_per_parcel(parcels, saliency, x_vals, y_vals, top_k=5):
    """
    parcels: GeoDataFrame of parcels
    saliency: [H, W] saliency map
    x_vals, y_vals: 1D arrays from NetCDF
    top_k: number of top pixels per parcel

    Returns: list of (x_pixel, y_pixel) top saliency pixels across all parcels
    """
    H, W = saliency.shape
    all_prompts = []

    for _, parcel in parcels.iterrows():
        geom = parcel.geometry
        if geom is None or geom.is_empty:
            continue

        # Find pixels inside the parcel
        min_x, min_y, max_x, max_y = geom.bounds
        # Approximate pixel range
        x_min_idx = np.searchsorted(x_vals, min_x) - 1
        x_max_idx = np.searchsorted(x_vals, max_x) + 1
        y_min_idx = np.searchsorted(y_vals, min_y) - 1
        y_max_idx = np.searchsorted(y_vals, max_y) + 1

        x_min_idx = max(0, x_min_idx)
        x_max_idx = min(W, x_max_idx)
        y_min_idx = max(0, y_min_idx)
        y_max_idx = min(H, y_max_idx)

        parcel_pixels = []
        for i in range(y_min_idx, y_max_idx):
            for j in range(x_min_idx, x_max_idx):
                # Check if pixel is inside parcel
                x_map = x_vals[j]
                y_map = y_vals[i]
                point = Point(x_map, y_map)
                if geom.contains(point):
                    parcel_pixels.append((saliency[i, j], j, i))  # (saliency, x, y)

        # Sort by saliency descending, take top_k
        parcel_pixels.sort(key=lambda x: x[0], reverse=True)
        top_pixels = parcel_pixels[:top_k]
        all_prompts.extend([(x, y) for _, x, y in top_pixels])

    return all_prompts


# -------------------------------------------------------
# SAVE CENTROIDS
# -------------------------------------------------------
def save_temporal_centroids(image_id, centroids):
    os.makedirs(TEMPORAL_PROMPTS_DIR, exist_ok=True)
    out_path = os.path.join(
        TEMPORAL_PROMPTS_DIR,
        f"superpixel_prompts_{image_id}.npy"
    )
    np.save(out_path, np.array(centroids))
    print(f"[Temporal SPX] Saved {len(centroids)} centroids → {out_path}")
    return out_path

# -------------------------------------------------------
# Run Temporal superpixels
# -------------------------------------------------------
def run_temporal_superpixels(dataset, country="Netherlands"):
    # Load parcel vector file once
    print("📥 Loading parcel vector file...")
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )
    big_gdf = big_gdf[big_gdf["coutry"] == country]
    big_gdf["parcel_id"] = range(len(big_gdf))

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    print("🚀 Generating bounded temporal superpixel prompts...\n")

    for item in dataset:
        image_id = item["id"]

        # Find corresponding NetCDF path
        nc_path = next((p for p in nc_files if extract_image_id(p) == image_id), None)
        if nc_path is None:
            print(f"SKIP: No NetCDF found for image {image_id}")
            continue

        # Load parcels for this image
        parcels = load_parcels(big_gdf, image_id, country)
        if len(parcels) == 0:
            print(f"SKIP: No parcels for image {image_id}")
            continue

        # Load NetCDF spatial axes
        ds = xr.open_dataset(nc_path)
        x_vals = ds["x"].values
        y_vals = ds["y"].values
        ds.close()  # Close to free memory

        # Compute saliency
        saliency = build_temporal_saliency_mask(item["ndvi_ts"])

        # Select top saliency pixels per parcel
        prompts = select_top_saliency_pixels_per_parcel(parcels, saliency, x_vals, y_vals, top_k=5)

        print(f"Image {image_id}: {len(parcels)} parcels → {len(prompts)} prompts (top 5 per parcel)")

        # Save prompts
        save_temporal_centroids(image_id, prompts)
