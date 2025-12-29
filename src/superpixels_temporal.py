import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.util import img_as_float


from src.config import TEMPORAL_PROMPTS_DIR, VECTOR_DIR, SPATIAL_PROMPTS_DIR, COMBINED_PROMPTS_DIR, ADAPTIVE_PROMPTS_DIR
from src.saliency_builders import (
    TemporalSaliencyBuilder,
    SpatialSaliencyBuilder, 
    CombinedSaliencyBuilder,
    AdaptiveSaliencyBuilder
)

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
# ENHANCED SALIENCY MASK BUILDERS (NEW)
# -------------------------------------------------------
def build_enhanced_saliency_mask(ndvi_ts, method='spatial', **kwargs):
    """
    Enhanced saliency mask builder with multiple methods.
    
    Args:
        ndvi_ts: NDVI time-series [T, H, W]
        method: 'temporal' | 'spatial' | 'adaptive' | 'combined'
        **kwargs: method-specific parameters
    
    Returns:
        saliency_mask: float32 [H, W] in [0, 1]
    """
    
    if method == 'temporal':
        # EXISTING BEHAVIOR - exact same as before
        return TemporalSaliencyBuilder.build(ndvi_ts, **kwargs)
    
    elif method == 'spatial':
        # NEW: spatial gradient based
        return SpatialSaliencyBuilder.build(ndvi_ts, **kwargs)
    
    elif method == 'adaptive':
        # NEW: automatically choose best method
        saliency, method_info = AdaptiveSaliencyBuilder.build(ndvi_ts, **kwargs)
        print(f"[Adaptive] Selected method: {method_info['selected_method']} "
              f"(confidence: {method_info['confidence']:.2f})")
        return saliency, method_info
    
    elif method == 'combined':
        # NEW: combine temporal + spatial
        return CombinedSaliencyBuilder.build(ndvi_ts, **kwargs)
    
    else:
        raise ValueError(f"Unknown saliency method: {method}")


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
# ENHANCED PROMPT SELECTION (NEW)
# -------------------------------------------------------
def select_enhanced_prompts_per_parcel(parcels, saliency, x_vals, y_vals, 
                                     top_k=5, selection_strategy='spatial_diversity', **kwargs):
    """
    Enhanced prompt selection with multiple strategies.
    
    Args:
        parcels: GeoDataFrame of parcels
        saliency: [H, W] saliency map
        x_vals, y_vals: 1D arrays from NetCDF
        top_k: number of top pixels per parcel
        selection_strategy: 'top_k' | 'spatial_diversity' | 'boundary_aligned'
    
    Returns:
        list of (x_pixel, y_pixel) selected prompts
    """
    
    if selection_strategy == 'top_k':
        # EXISTING BEHAVIOR
        return select_top_saliency_pixels_per_parcel(parcels, saliency, x_vals, y_vals, top_k)
    
    elif selection_strategy == 'spatial_diversity':
        # NEW: ensures spatial spread across parcel
        return _select_spatial_diversity_prompts(parcels, saliency, x_vals, y_vals, top_k, **kwargs)
    
    elif selection_strategy == 'boundary_aligned':
        # NEW: follows parcel boundaries systematically
        return _select_boundary_aligned_prompts(parcels, saliency, x_vals, y_vals, top_k, **kwargs)
    
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")


def _select_spatial_diversity_prompts(parcels, saliency, x_vals, y_vals, top_k, min_distance=10):
    """Select prompts ensuring minimum spatial distance between them."""
    H, W = saliency.shape
    all_prompts = []
    
    for _, parcel in parcels.iterrows():
        geom = parcel.geometry
        if geom is None or geom.is_empty:
            continue
        
        # Find pixels inside the parcel
        min_x, min_y, max_x, max_y = geom.bounds
        x_min_idx = max(0, np.searchsorted(x_vals, min_x) - 1)
        x_max_idx = min(W, np.searchsorted(x_vals, max_x) + 1)
        y_min_idx = max(0, np.searchsorted(y_vals, min_y) - 1)
        y_max_idx = min(H, np.searchsorted(y_vals, max_y) + 1)
        
        # Collect all valid pixels
        valid_pixels = []
        for i in range(y_min_idx, y_max_idx):
            for j in range(x_min_idx, x_max_idx):
                x_map = x_vals[j]
                y_map = y_vals[i]
                point = Point(x_map, y_map)
                if geom.contains(point):
                    valid_pixels.append((saliency[i, j], j, i))
        
        # Sort by saliency
        valid_pixels.sort(key=lambda x: x[0], reverse=True)
        
        # Select diverse prompts
        selected_prompts = []
        for sal_val, x, y in valid_pixels:
            # Check minimum distance from already selected prompts
            too_close = False
            for sel_x, sel_y in selected_prompts:
                distance = np.sqrt((x - sel_x)**2 + (y - sel_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_prompts.append((x, y))
                if len(selected_prompts) >= top_k:
                    break
        
        all_prompts.extend(selected_prompts)
    
    return all_prompts


def _select_boundary_aligned_prompts(parcels, saliency, x_vals, y_vals, top_k, boundary_density=0.3):
    """Select prompts aligned with parcel boundaries."""
    H, W = saliency.shape
    all_prompts = []
    
    for _, parcel in parcels.iterrows():
        geom = parcel.geometry
        if geom is None or geom.is_empty:
            continue
        
        # Find boundary pixels (high gradient areas near parcel edges)
        boundary_prompts = _find_boundary_prompts(geom, saliency, x_vals, y_vals, top_k)
        all_prompts.extend(boundary_prompts)
    
    return all_prompts


def _find_boundary_prompts(geom, saliency, x_vals, y_vals, top_k):
    """Find prompts specifically along parcel boundaries."""
    # This is a simplified implementation
    # In practice, you might want to use more sophisticated boundary detection
    prompts = []
    
    # For now, use the existing top-k approach but focus on boundary regions
    min_x, min_y, max_x, max_y = geom.bounds
    
    # Focus on edges of the parcel (first and last 20% of pixels)
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    edge_threshold = 0.2
    
    # Get parcel bounds in pixel coordinates
    x_min_idx = max(0, np.searchsorted(x_vals, min_x) - 1)
    x_max_idx = min(len(x_vals)-1, np.searchsorted(x_vals, max_x) + 1)
    y_min_idx = max(0, np.searchsorted(y_vals, min_y) - 1)
    y_max_idx = min(len(y_vals)-1, np.searchsorted(y_vals, max_y) + 1)
    
    # Collect edge pixels with high saliency
    edge_pixels = []
    
    for i in range(y_min_idx, y_max_idx):
        for j in range(x_min_idx, x_max_idx):
            x_map = x_vals[j]
            y_map = y_vals[i]
            point = Point(x_map, y_map)
            
            if geom.contains(point):
                # Check if pixel is near parcel edge
                x_norm = (x_map - min_x) / x_range if x_range > 0 else 0.5
                y_norm = (y_map - min_y) / y_range if y_range > 0 else 0.5
                
                is_edge = (x_norm < edge_threshold or x_norm > 1-edge_threshold or
                          y_norm < edge_threshold or y_norm > 1-edge_threshold)
                
                if is_edge:
                    edge_pixels.append((saliency[i, j], j, i))
    
    # Select top saliency edge pixels
    edge_pixels.sort(key=lambda x: x[0], reverse=True)
    selected_edges = edge_pixels[:top_k]
    
    return [(x, y) for _, x, y in selected_edges]


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
# NEW SPATIAL PROMPT SAVING FUNCTIONS (NEW)
# -------------------------------------------------------
def save_spatial_centroids(image_id, centroids):
    """Save spatial gradient-based prompts"""
    os.makedirs(SPATIAL_PROMPTS_DIR, exist_ok=True)
    out_path = os.path.join(
        SPATIAL_PROMPTS_DIR,
        f"spatial_prompts_{image_id}.npy"
    )
    np.save(out_path, np.array(centroids))
    print(f"[Spatial SPX] Saved {len(centroids)} centroids → {out_path}")
    return out_path


def save_combined_centroids(image_id, centroids):
    """Save combined temporal + spatial prompts"""
    os.makedirs(COMBINED_PROMPTS_DIR, exist_ok=True)
    out_path = os.path.join(
        COMBINED_PROMPTS_DIR,
        f"combined_prompts_{image_id}.npy"
    )
    np.save(out_path, np.array(centroids))
    print(f"[Combined SPX] Saved {len(centroids)} centroids → {out_path}")
    return out_path


def save_adaptive_centroids(image_id, centroids, method_info):
    """Save adaptive method prompts with metadata"""
    os.makedirs(ADAPTIVE_PROMPTS_DIR, exist_ok=True)
    out_path = os.path.join(
        ADAPTIVE_PROMPTS_DIR,
        f"adaptive_prompts_{image_id}.npy"
    )
    
    # Save prompts and metadata
    np.save(out_path, np.array(centroids))
    
    # Save method info as separate file
    info_path = out_path.replace('.npy', '_method_info.npy')
    np.save(info_path, method_info)
    
    print(f"[Adaptive SPX] Saved {len(centroids)} centroids → {out_path}")
    print(f"[Adaptive SPX] Method info → {info_path}")
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


# -------------------------------------------------------
# LEGACY COMPATIBILITY (Maintains existing function signature)
# -------------------------------------------------------
def run_spatial_superpixels_simple(dataset, country="Netherlands"):
    """Legacy compatibility - calls new implementation with defaults"""
    return run_spatial_superpixels(dataset, country, use_cuda=True, top_k=5)


# -------------------------------------------------------
# NEW SPATIAL GRADIENT PIPELINES (NEW)
# -------------------------------------------------------
def run_spatial_superpixels(dataset, country="Netherlands", use_cuda=True, top_k=5):
    """Generate spatial gradient-based prompts"""
    # Load parcel vector file once
    print("📥 Loading parcel vector file...")
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )
    big_gdf = big_gdf[big_gdf["coutry"] == country]
    big_gdf["parcel_id"] = range(len(big_gdf))

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    print("🚀 Generating spatial gradient superpixel prompts...\n")

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

        # Compute spatial saliency using gradients
        print(f"[Spatial] Computing gradients for image {image_id}...")
        saliency = build_enhanced_saliency_mask(
            item["ndvi_ts"], 
            method='spatial',
            use_cuda=use_cuda
        )

        # Select spatially diverse prompts
        prompts = select_enhanced_prompts_per_parcel(
            parcels, saliency, x_vals, y_vals, 
            top_k=top_k,
            selection_strategy='spatial_diversity'
        )

        print(f"Image {image_id}: {len(parcels)} parcels → {len(prompts)} prompts "
              f"(spatial diversity, top {top_k} per parcel)")

        # Save spatial prompts
        save_spatial_centroids(image_id, prompts)


def run_combined_superpixels(dataset, country="Netherlands", use_cuda=True, top_k=5):
    """Generate combined temporal + spatial prompts"""
    # Load parcel vector file once
    print("📥 Loading parcel vector file...")
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )
    big_gdf = big_gdf[big_gdf["coutry"] == country]
    big_gdf["parcel_id"] = range(len(big_gdf))

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    print("🚀 Generating combined superpixel prompts...\n")

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

        # Compute combined saliency
        print(f"[Combined] Computing combined saliency for image {image_id}...")
        saliency = build_enhanced_saliency_mask(
            item["ndvi_ts"], 
            method='combined',
            use_cuda=use_cuda
        )

        # Select prompts with spatial diversity
        prompts = select_enhanced_prompts_per_parcel(
            parcels, saliency, x_vals, y_vals, 
            top_k=top_k,
            selection_strategy='spatial_diversity'
        )

        print(f"Image {image_id}: {len(parcels)} parcels → {len(prompts)} prompts "
              f"(combined method, top {top_k} per parcel)")

        # Save combined prompts
        save_combined_centroids(image_id, prompts)


def run_adaptive_superpixels(dataset, country="Netherlands", use_cuda=True, top_k=5):
    """Generate adaptive prompts with automatic method selection"""
    # Load parcel vector file once
    print("📥 Loading parcel vector file...")
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )
    big_gdf = big_gdf[big_gdf["coutry"] == country]
    big_gdf["parcel_id"] = range(len(big_gdf))

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    print("🚀 Generating adaptive superpixel prompts...\n")

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

        # Compute adaptive saliency
        print(f"[Adaptive] Computing adaptive saliency for image {image_id}...")
        saliency, method_info = build_enhanced_saliency_mask(
            item["ndvi_ts"], 
            method='adaptive',
            preference='boundary_detection',
            use_cuda=use_cuda
        )

        # Select prompts with spatial diversity
        prompts = select_enhanced_prompts_per_parcel(
            parcels, saliency, x_vals, y_vals, 
            top_k=top_k,
            selection_strategy='spatial_diversity'
        )

        print(f"Image {image_id}: {len(parcels)} parcels → {len(prompts)} prompts "
              f"(adaptive method, top {top_k} per parcel)")

        # Save adaptive prompts with metadata
        save_adaptive_centroids(image_id, prompts, method_info)


def run_all_superpixel_methods(dataset, country="Netherlands", use_cuda=True, top_k=5):
    """Run all prompt generation methods and save results"""
    
    print("🚀 Running ALL superpixel generation methods...\n")
    
    # Method 1: Temporal (existing)
    print("=" * 60)
    print("METHOD 1: TEMPORAL (Existing Implementation)")
    print("=" * 60)
    run_temporal_superpixels(dataset, country)
    
    # Method 2: Spatial (new)
    print("\n" + "=" * 60)
    print("METHOD 2: SPATIAL GRADIENTS (New Implementation)")
    print("=" * 60)
    run_spatial_superpixels(dataset, country, use_cuda, top_k)
    
    # Method 3: Combined (new)
    print("\n" + "=" * 60)
    print("METHOD 3: COMBINED TEMPORAL + SPATIAL (New Implementation)")
    print("=" * 60)
    run_combined_superpixels(dataset, country, use_cuda, top_k)
    
    # Method 4: Adaptive (new)
    print("\n" + "=" * 60)
    print("METHOD 4: ADAPTIVE (Auto-selects Best Method)")
    print("=" * 60)
    run_adaptive_superpixels(dataset, country, use_cuda, top_k)
    
    print("\n✅ ALL superpixel generation methods completed!")
    print("\nResults saved to:")
    print(f"  - Temporal: {TEMPORAL_PROMPTS_DIR}")
    print(f"  - Spatial:  {SPATIAL_PROMPTS_DIR}")
    print(f"  - Combined: {COMBINED_PROMPTS_DIR}")
    print(f"  - Adaptive: {ADAPTIVE_PROMPTS_DIR}")
