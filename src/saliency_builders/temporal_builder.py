# src/saliency_builders/temporal_builder.py
import sys
import os

# Add project root to Python path
# This goes up 3 levels: saliency_builders/ -> src/ -> project_root/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from src.ndvi_metrics import CombinedTemporalMetric

class TemporalSaliencyBuilder:
    """
    Saliency builder that replicates the existing temporal approach.
    Uses NDVI variance + peak difference.
    """
    
    @staticmethod
    def build(ndvi_ts: np.ndarray, var_thresh=0.05, peak_thresh=0.15) -> np.ndarray:
        """
        Build temporal saliency mask (replicates existing functionality).
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            var_thresh: variance threshold
            peak_thresh: peak difference threshold
            
        Returns:
            saliency_mask: float32 [H, W] in [0, 1]
        """
        # Use the combined temporal metric
        metric = CombinedTemporalMetric()
        saliency = metric.compute(ndvi_ts)
        
        # Apply thresholds (similar to existing implementation)
        def norm(x):
            x = x.astype(np.float32)
            return (x - x.min()) / (x.max() - x.min() + 1e-6)
        
        # Create binary masks
        var_mask = (saliency > var_thresh).astype(np.float32)
        peak_mask = (saliency > peak_thresh).astype(np.float32)
        
        # Combine (OR-like)
        combined = np.clip(var_mask + peak_mask, 0, 1)
        
        return combined.astype(np.float32)
