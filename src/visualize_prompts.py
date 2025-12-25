# src/visualize_prompts.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import glob
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rasterio
from shapely.geometry import Point

from src.config import (
    IMAGES_DIR,
    VECTOR_DIR,
    VANILLA_PROMPTS_DIR,
    TEMPORAL_PROMPTS_DIR,
)
from src.step2_scaling import collect_image_paths, load_nc_image, load_parcels, extract_image_id


def load_parcels_for_image(big_gdf, image_id, country="Netherlands"):
    """
    Load and filter parcels for a specific image.
    Returns GeoDataFrame of parcels, or empty if none found.
    """
    parcels = load_parcels(big_gdf, image_id, country)
    if len(parcels) == 0:
        print(f"SKIP: No parcels for image {image_id}")
    return parcels


def compute_parcel_coverage(prompts, parcels, x_vals, y_vals):
    """
    Compute the percentage of prompts that fall within any parcel.
    prompts: np.array of shape (N, 2) with (x, y) pixel coords
    parcels: GeoDataFrame
    x_vals, y_vals: 1D arrays from NetCDF for coord conversion
    Returns: float percentage (0-100)
    """
    if len(prompts) == 0:
        return 0.0

    inside_count = 0
    for x_pixel, y_pixel in prompts:
        x_map = x_vals[int(x_pixel)]
        y_map = y_vals[int(y_pixel)]
        point = Point(x_map, y_map)
        for _, parcel in parcels.iterrows():
            if parcel.geometry.contains(point):
                inside_count += 1
                break

    return (inside_count / len(prompts)) * 100.0


def _lock_imshow_limits(ax, rgb):
    """Keep image visible even after GeoPandas plots autoscale the axes.

    GeoPandas' `.plot()` updates x/y limits to geometry coordinates (often in
    projected CRS units), which can make the pixel-based image disappear.
    """
    h, w = rgb.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def _plot_parcel_boundaries_pixels(ax, parcels_gdf, mask_path, rgb, *, color="yellow", linewidth=1):
    """Plot parcel boundaries in *pixel coordinates* on top of an image.

    Assumptions (per project setup):
    - `parcels_gdf` geometries are in the same CRS as the GeoTIFF at `mask_path`.
    - The mask GeoTIFF's affine transform maps world coordinates -> raster pixels.

    We convert world (x,y) -> pixel (col,row) using the inverse affine transform,
    then (optionally) scale to the RGB array resolution if it differs from mask.
    """
    with rasterio.open(mask_path) as src:
        inv = ~src.transform
        raster_h, raster_w = src.height, src.width

    img_h, img_w = rgb.shape[:2]
    sx = (img_w / raster_w) if raster_w else 1.0
    sy = (img_h / raster_h) if raster_h else 1.0

    def _iter_lines(geom):
        if geom is None or geom.is_empty:
            return
        boundary = geom.boundary
        gtype = getattr(boundary, "geom_type", None)
        if gtype == "LineString":
            yield boundary
        elif gtype == "MultiLineString":
            for line in boundary.geoms:
                yield line

    for geom in getattr(parcels_gdf, "geometry", []):
        for line in _iter_lines(geom):
            coords = np.asarray(line.coords)
            if coords.size == 0:
                continue
            xs, ys = coords[:, 0], coords[:, 1]

            # world -> pixel (col,row)
            cols, rows = inv * (xs, ys)
            cols = np.asarray(cols) * sx
            rows = np.asarray(rows) * sy

            ax.plot(cols, rows, color=color, linewidth=linewidth, zorder=3)


def visualize_vanilla_prompts(dataset):
    # Load parcels once
    big_gdf = gpd.read_file(os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg"))
    big_gdf = big_gdf[big_gdf["coutry"] == "Netherlands"]

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    for item in dataset:
        image_id = item["id"]

        # Load prompts
        prompts_path = os.path.join(VANILLA_PROMPTS_DIR, f"vanilla_prompts_{image_id}.npy")
        if not os.path.exists(prompts_path):
            continue
        prompts = np.load(prompts_path)
        print(f"Loaded {len(prompts)} temporal prompts for image {image_id}")

        # Load parcels
        parcels = load_parcels_for_image(big_gdf, image_id)
        if len(parcels) == 0:
            continue

        # Use RGB from dataset
        rgb = item["rgb"]
        rgb = (rgb * 255).astype(np.uint8)

        # Load NetCDF for x/y vals (for parcel plotting)
        nc_path = next((p for p in nc_files if extract_image_id(p) == image_id), None)
        if nc_path is None:
            continue
        ds = xr.open_dataset(nc_path)
        x_vals = ds.x.values
        y_vals = ds.y.values
        ds.close()  # Close to free memory

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(rgb)
        _lock_imshow_limits(ax, rgb)

        # Plot parcels
        _plot_parcel_boundaries_pixels(
            ax,
            parcels,
            item["mask_path"],
            rgb,
            color="red",
            linewidth=2,
        )

        # Plot prompts
        if len(prompts) > 0:
            ax.scatter(prompts[:, 0], prompts[:, 1], c='red', s=60, label='Vanilla Prompts')

        _lock_imshow_limits(ax, rgb)
        ax.set_title(f"Image {image_id} - Vanilla Prompts")
        ax.axis('off')
        plt.legend()
        plt.show()


def visualize_vanilla_vs_temporal(dataset, country="Netherlands"):
    """
    Visualize:
      - RGB image
      - Parcel boundary
      - Vanilla centroid prompts (red)
      - Temporal ranked prompts (blue)

    Two columns:
      LEFT  = Vanilla
      RIGHT = Temporal
    """

    # --------------------------------------------------
    # Load parcels once
    # --------------------------------------------------
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    shown = 0

    for item in dataset:
        image_id = item["id"]

        vanilla_path = os.path.join(
            VANILLA_PROMPTS_DIR, f"vanilla_prompts_{image_id}.npy"
        )
        temporal_path = os.path.join(
            TEMPORAL_PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy"
        )

        # ---- Skip if prompts missing ----
        if not os.path.exists(vanilla_path) or not os.path.exists(temporal_path):
            continue

        parcels = load_parcels_for_image(big_gdf, image_id, country)
        if len(parcels) == 0:
            continue

        # Use RGB from dataset
        rgb = item["rgb"]
        rgb = (rgb * 255).astype(np.uint8)

        # Load NetCDF for x/y vals
        nc_path = next((p for p in nc_files if extract_image_id(p) == image_id), None)
        if nc_path is None:
            continue
        ds = xr.open_dataset(nc_path)
        x_vals = ds.x.values
        y_vals = ds.y.values
        ds.close()  # Close to free memory

        vanilla_pts = np.load(vanilla_path)
        temporal_pts = np.load(temporal_path)

        if len(vanilla_pts) == 0 or len(temporal_pts) == 0:
            continue

        # Compute parcel coverage
        vanilla_coverage = compute_parcel_coverage(vanilla_pts, parcels, x_vals, y_vals)
        temporal_coverage = compute_parcel_coverage(temporal_pts, parcels, x_vals, y_vals)
        print(f"Image {image_id}: Vanilla coverage {vanilla_coverage:.1f}%, Temporal coverage {temporal_coverage:.1f}%")

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # ---------- Vanilla ----------
        axes[0].imshow(rgb)
        _lock_imshow_limits(axes[0], rgb)
        _plot_parcel_boundaries_pixels(
            axes[0],
            parcels,
            item["mask_path"],
            rgb,
            color="red",
            linewidth=2,
        )
        axes[0].scatter(
            vanilla_pts[:, 0],
            vanilla_pts[:, 1],
            c="red",
            s=60,
            label="Vanilla Centroids"
        )
        _lock_imshow_limits(axes[0], rgb)
        axes[0].set_title(f"Vanilla — Image {image_id}")
        axes[0].axis("off")

        # ---------- Temporal ----------
        axes[1].imshow(rgb)
        _lock_imshow_limits(axes[1], rgb)
        _plot_parcel_boundaries_pixels(
            axes[1],
            parcels,
            item["mask_path"],
            rgb,
            color="red",
            linewidth=2,
        )
        axes[1].scatter(
            temporal_pts[:, 0],
            temporal_pts[:, 1],
            c="blue",
            s=20,
            alpha=0.7,
            label="Temporal Prompts"
        )
        _lock_imshow_limits(axes[1], rgb)
        axes[1].set_title(f"Temporal — Image {image_id}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        shown += 1

    print(f"✅ Visualized {shown} images.")


def visualize_vanilla_and_temporal_prompts(dataset, country="Netherlands"):
    """
    Visualize vanilla prompts + top 5 temporal prompts + parcel boundaries.
    Ensures prompts are bounded within parcels.
    """

    # --------------------------------------------------
    # Load parcels once
    # --------------------------------------------------
    big_gdf = gpd.read_file(
        os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg")
    )

    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    shown = 0

    for item in dataset:
        image_id = item["id"]

        vanilla_path = os.path.join(
            VANILLA_PROMPTS_DIR, f"vanilla_prompts_{image_id}.npy"
        )
        temporal_path = os.path.join(
            TEMPORAL_PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy"
        )

        # ---- Skip if prompts missing ----
        if not os.path.exists(vanilla_path) or not os.path.exists(temporal_path):
            continue

        parcels = load_parcels_for_image(big_gdf, image_id, country)
        if len(parcels) == 0:
            continue

        # ---- Load image ----
        # load_nc_image from step2_scaling returns (ndvi, rgb), so we only need rgb
        _, rgb = load_nc_image(item["nc_path"])
        rgb = (rgb * 255).astype(np.uint8)
        
        # Load NetCDF for coordinate values
        ds = xr.open_dataset(item["nc_path"])
        x_vals = ds.x.values
        y_vals = ds.y.values
        ds.close()  # Close to free memory

        vanilla_pts = np.load(vanilla_path)
        temporal_pts = np.load(temporal_path)

        if len(vanilla_pts) == 0 or len(temporal_pts) == 0:
            continue

        # Compute parcel coverage
        vanilla_coverage = compute_parcel_coverage(vanilla_pts, parcels, x_vals, y_vals)
        temporal_coverage = compute_parcel_coverage(temporal_pts, parcels, x_vals, y_vals)
        print(f"Image {image_id}: Vanilla coverage {vanilla_coverage:.1f}%, Temporal coverage {temporal_coverage:.1f}%")

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(rgb)
        _lock_imshow_limits(ax, rgb)

        # Plot parcels
        _plot_parcel_boundaries_pixels(
            ax,
            parcels,
            item["mask_path"],
            rgb,
            color="yellow",
            linewidth=2,
        )

        # Plot vanilla prompts
        ax.scatter(
            vanilla_pts[:, 0],
            vanilla_pts[:, 1],
            c="red",
            s=60,
            marker="o",
            label="Vanilla Prompts",
            zorder=5
        )

        # Plot temporal prompts
        ax.scatter(
            temporal_pts[:, 0],
            temporal_pts[:, 1],
            c="blue",
            s=30,
            marker="x",
            alpha=0.8,
            label="Temporal Prompts (Top 5 per parcel)",
            zorder=5
        )

        ax.set_title(f"Image {image_id} - Vanilla + Temporal Prompts (Bounded)")
        ax.axis('off')
        plt.legend()
        plt.show()

        shown += 1

    print(f"✅ Visualized {shown} images.")


#-------------------------------------------------------
def visualize_temporal_prompts(dataset):
    # Load parcels once
    big_gdf = gpd.read_file(os.path.join(VECTOR_DIR, "ai4boundaries_parcels_vector_sampled.gpkg"))
    big_gdf = big_gdf[big_gdf["coutry"] == "Netherlands"]
    
    dataset = dataset
    # Get NetCDF paths
    nc_files, _ = collect_image_paths()

    for item in dataset:
        image_id = item["id"]
        
        # Load prompts
        prompts_path = os.path.join(TEMPORAL_PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy")
        if not os.path.exists(prompts_path):
            continue
        prompts = np.load(prompts_path)
        
        # Load parcels
        parcels = load_parcels_for_image(big_gdf, image_id)
        if len(parcels) == 0:
            continue
        
        # Use RGB from dataset
        rgb = item["rgb"]
        rgb = (rgb * 255).astype(np.uint8)
        
        # Load NetCDF for x/y vals (for parcel plotting)
        nc_path = next((p for p in nc_files if extract_image_id(p) == image_id), None)
        if nc_path is None:
            continue
        ds = xr.open_dataset(nc_path)
        x_vals = ds.x.values
        y_vals = ds.y.values
        ds.close()  # Close to free memory
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(rgb)
        _lock_imshow_limits(ax, rgb)

        # Plot parcels
        _plot_parcel_boundaries_pixels(
            ax,
            parcels,
            item["mask_path"],
            rgb,
            color="red",
            linewidth=2,
        )

        # Plot prompts
        if len(prompts) > 0:
            ax.scatter(prompts[:, 0], prompts[:, 1], c='blue', s=50, alpha=0.8, zorder=4, label='Temporal Prompts')

        _lock_imshow_limits(ax, rgb)
        ax.set_title(f"Image {image_id} - Bounded Temporal Prompts")
        ax.axis('off')
        plt.legend()
        plt.show()
    