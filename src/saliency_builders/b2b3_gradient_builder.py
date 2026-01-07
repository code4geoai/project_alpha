# src/saliency_builders/b2b3_gradient_builder.py
import sys
import os

# Add project root to Python path
# This goes up 3 levels: saliency_builders/ -> src/ -> project_root/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from skimage.filters import sobel

class B2B3GradientSaliencyBuilder:
    """
    Saliency builder for B2-B3 gradient analysis.
    Computes Sobel gradients for blue (B2) and green (B3) bands,
    combines them by averaging, and applies thresholds.
    """

    @staticmethod
    def build(b2_ts: np.ndarray, b3_ts: np.ndarray, grad_thresh=0.1) -> np.ndarray:
        """
        Build B2-B3 gradient saliency mask.

        Args:
            b2_ts: B2 time-series [T, H, W] or single band [H, W]
            b3_ts: B3 time-series [T, H, W] or single band [H, W]
            grad_thresh: gradient threshold for saliency

        Returns:
            saliency_mask: float32 [H, W] in [0, 1]
        """
        # Handle time-series: use median or first timestep
        if b2_ts.ndim == 3:
            b2 = np.median(b2_ts, axis=0)
            b3 = np.median(b3_ts, axis=0)
        else:
            b2 = b2_ts
            b3 = b3_ts

        # Compute Sobel gradients for each band
        grad_b2 = sobel(b2.astype(np.float32))
        grad_b3 = sobel(b3.astype(np.float32))

        # Combine by simple averaging
        combined_grad = (grad_b2 + grad_b3) / 2.0

        # Normalize to [0, 1]
        grad_min, grad_max = combined_grad.min(), combined_grad.max()
        if grad_max > grad_min:
            saliency = (combined_grad - grad_min) / (grad_max - grad_min)
        else:
            saliency = np.zeros_like(combined_grad)

        # Apply threshold
        saliency = np.where(saliency > grad_thresh, saliency, 0)

        return saliency.astype(np.float32)