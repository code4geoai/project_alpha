# src/saliency_builders/spatial_builder.py

import numpy as np
from ndvi_metrics import CombinedSpatialMetric, SobelGradientMetric

class SpatialSaliencyBuilder:
    """
    Spatial saliency builder using NDVI spatial gradients.
    Primary approach for agricultural boundary detection.
    """
    
    @staticmethod
    def build(ndvi_ts: np.ndarray, 
              use_cuda=True,
              gradient_threshold=0.1,
              return_components=False) -> np.ndarray:
        """
        Build spatial saliency mask using gradient-based boundary detection.
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            use_cuda: whether to use CUDA acceleration
            gradient_threshold: threshold for gradient magnitude
            return_components: whether to return individual components
            
        Returns:
            saliency_mask: float32 [H, W] in [0, 1]
            components: dict of individual metric outputs (if return_components=True)
        """
        # Primary spatial gradient metric
        gradient_metric = SobelGradientMetric(
            use_median=True,
            gradient_threshold=gradient_threshold,
            use_cuda=use_cuda
        )
        
        # Combined spatial metric for robust boundaries
        spatial_metric = CombinedSpatialMetric(
            gradient_weight=0.6,
            variance_weight=0.2, 
            quantile_weight=0.2,
            use_cuda=use_cuda
        )
        
        # Compute spatial saliency
        spatial_saliency = spatial_metric.compute(ndvi_ts)
        
        if return_components:
            # Return individual components for analysis
            components = {
                'gradient': gradient_metric.compute(ndvi_ts),
                'variance': spatial_metric.variance_metric.compute(ndvi_ts),
                'quantile': spatial_metric.quantile_metric.compute(ndvi_ts),
                'combined': spatial_saliency
            }
            return spatial_saliency, components
        else:
            return spatial_saliency
    
    @staticmethod
    def build_adaptive_threshold(ndvi_ts: np.ndarray, 
                                use_cuda=True,
                                percentile_threshold=80) -> np.ndarray:
        """
        Build spatial saliency with adaptive threshold selection.
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            use_cuda: whether to use CUDA acceleration
            percentile_threshold: percentile for adaptive threshold
            
        Returns:
            saliency_mask: float32 [H, W] with adaptive thresholding
        """
        # Compute gradient without threshold
        gradient_metric = SobelGradientMetric(
            use_median=True,
            gradient_threshold=0.0,  # No fixed threshold
            use_cuda=use_cuda
        )
        
        gradient_map = gradient_metric.compute(ndvi_ts)
        
        # Adaptive threshold based on percentile
        threshold = np.percentile(gradient_map, percentile_threshold)
        
        # Apply adaptive threshold
        adaptive_saliency = np.maximum(gradient_map - threshold, 0)
        
        # Normalize
        if np.max(adaptive_saliency) > 0:
            adaptive_saliency = adaptive_saliency / np.max(adaptive_saliency)
        
        return adaptive_saliency
