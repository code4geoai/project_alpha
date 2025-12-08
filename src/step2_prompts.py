import numpy as np
from skimage.feature import canny
from shapely.geometry import Point
import random

# ---------------------------
# 1. NDVI-variance prompts
# ---------------------------
def sample_ndvi_variance_points(ndvi_var_mask, stride=4):
    points = []
    H, W = ndvi_var_mask.shape

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            if ndvi_var_mask[y, x] == 1:
                points.append([x, y])
    return points


# ---------------------------
# 2. NDVI-edge prompts
# ---------------------------
def sample_edge_points(ndvi_mean, percentile=85):
    # gradient magnitude via Canny
    edge_map = canny(ndvi_mean, sigma=1.4)
    
    # convert boolean edge map → point list
    ys, xs = np.where(edge_map == 1)
    pts = list(zip(xs, ys))

    # keep strongest top percentile edges
    if len(pts) == 0:
        return []

    k = int(len(pts) * (percentile / 100))
    return pts[:k]


# ---------------------------
# 3. Parcel-based prompts
# ---------------------------
def sample_parcel_points(parcels, ndvi_shape, points_per_parcel=1):
    H, W = ndvi_shape
    pts = []

    for _, row in parcels.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue

        for _ in range(points_per_parcel):
            # random point inside polygon
            minx, miny, maxx, maxy = geom.bounds
            
            for _ in range(20):
                px = random.uniform(minx, maxx)
                py = random.uniform(miny, maxy)
                p = Point(px, py)
                if geom.contains(p):
                    # convert geospatial → raster pixel
                    # (Assumes affine transform stored in attrs)
                    # user will supply later; for now placeholder
                    pass

        # NOTE: We will fill in accurate pixel mapping in 2E
    return pts


# ---------------------------
# 4. Combine all prompts
# ---------------------------
def build_prompts_for_image(entry, var_mask):
    ndvi_mean = entry["ndvi_mean"]
    parcels   = entry["parcels"]

    prompts_var = sample_ndvi_variance_points(var_mask, stride=4)
    prompts_edge = sample_edge_points(ndvi_mean, percentile=85)

    # parcels → pixel mapping temporarily skipped
    prompts_parcel = []

    return {
        "image_id": entry["id"],
        "var_points": prompts_var,
        "edge_points": prompts_edge,
        "parcel_points": prompts_parcel,
        "total": len(prompts_var) + len(prompts_edge) + len(prompts_parcel),
    }
