# src/saliency_builders/evi2_temporal_builder.py
import sys
import os

# Add project root to Python path
# This goes up 3 levels: saliency_builders/ -> src/ -> project_root/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from src.ndvi_metrics import CombinedEVITemporalMetric

class EVI2TemporalSaliencyBuilder:
    """
    Saliency builder for EVI2 temporal analysis.
    Uses EVI2 variance + peak difference.
    """

    @staticmethod
    def build(evi2_ts: np.ndarray, var_thresh=0.05, peak_thresh=0.15) -> np.ndarray:
        """
        Build EVI2 temporal saliency mask.

        Args:
            evi2_ts: EVI2 time-series [T, H, W]
            var_thresh: variance threshold
            peak_thresh: peak difference threshold

        Returns:
            saliency_mask: float32 [H, W] in [0, 1]
        """
        # Use the combined EVI2 temporal metric
        metric = CombinedEVITemporalMetric()
        saliency = metric.compute(evi2_ts)

        # Apply thresholds (similar to NDVI implementation)
        def norm(x):
            x = x.astype(np.float32)
            return (x - x.min()) / (x.max() - x.min() + 1e-6)

        # Create binary masks
        var_mask = (saliency > var_thresh).astype(np.float32)
        peak_mask = (saliency > peak_thresh).astype(np.float32)

        # Combine (OR-like)
        combined = np.clip(var_mask + peak_mask, 0, 1)

        return combined.astype(np.float32)