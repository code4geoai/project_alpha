import numpy as np

def select_best_ndvi_timestamp(ndvi_stack):
    """Return index of timestamp with maximum STD (best contrast)."""
    stds = np.nanstd(ndvi_stack, axis=(1,2))
    best_t = int(np.argmax(stds))
    return best_t, stds

def compute_slic_superpixels(image, n_segments=70):
    from skimage.segmentation import slic
    seg = slic(
        image,
        n_segments=n_segments,
        compactness=10,
        start_label=1,
        channel_axis=None
    )

    return seg