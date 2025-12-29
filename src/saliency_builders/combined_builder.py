# src/saliency_builders/combined_builder.py

import numpy as np
from ndvi_metrics import CombinedTemporalMetric, CombinedSpatialMetric

class CombinedSaliencyBuilder:
    """
    Combined saliency builder that fuses temporal and spatial approaches.
    """
    
    @staticmethod
    def build(ndvi_ts: np.ndarray,
              temporal_weight=0.3,
              spatial_weight=0.7,
              use_cuda=True) -> np.ndarray:
        """
        Build combined saliency mask by fusing temporal and spatial approaches.
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            temporal_weight: weight for temporal approach
            spatial_weight: weight for spatial approach
            use_cuda: whether to use CUDA acceleration
            
        Returns:
            combined_saliency: float32 [H, W] in [0, 1]
        """
        # Temporal metric
        temporal_metric = CombinedTemporalMetric(
            var_weight=0.5, 
            peak_weight=0.5
        )
        
        # Spatial metric
        spatial_metric = CombinedSpatialMetric(
            gradient_weight=0.6,
            variance_weight=0.2,
            quantile_weight=0.2,
            use_cuda=use_cuda
        )
        
        # Compute both saliency maps
        temporal_saliency = temporal_metric.compute(ndvi_ts)
        spatial_saliency = spatial_metric.compute(ndvi_ts)
        
        # Normalize weights
        total_weight = temporal_weight + spatial_weight
        temporal_weight = temporal_weight / total_weight
        spatial_weight = spatial_weight / total_weight
        
        # Fuse saliency maps
        combined_saliency = (temporal_weight * temporal_saliency + 
                           spatial_weight * spatial_saliency)
        
        # Final normalization
        combined_saliency = combined_saliency.astype(np.float32)
        min_val = np.min(combined_saliency)
        max_val = np.max(combined_saliency)
        
        if max_val - min_val > 1e-6:
            combined_saliency = (combined_saliency - min_val) / (max_val - min_val)
        else:
            combined_saliency = np.zeros_like(combined_saliency)
        
        return combined_saliency
    
    @staticmethod
    def build_adaptive_weights(ndvi_ts: np.ndarray,
                              use_cuda=True,
                              field_size_hints=None) -> np.ndarray:
        """
        Build combined saliency with adaptive weight selection.
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            use_cuda: whether to use CUDA acceleration
            field_size_hints: hints about field sizes for weight adaptation
            
        Returns:
            adaptive_saliency: float32 [H, W] with adaptive weights
        """
        # Compute temporal and spatial saliency
        temporal_metric = CombinedTemporalMetric()
        spatial_metric = CombinedSpatialMetric(use_cuda=use_cuda)
        
        temporal_saliency = temporal_metric.compute(ndvi_ts)
        spatial_saliency = spatial_metric.compute(ndvi_ts)
        
        # Adaptive weight selection based on saliency characteristics
        temporal_var = np.var(temporal_saliency)
        spatial_var = np.var(spatial_saliency)
        
        # Weight based on variance (more varied saliency gets higher weight)
        temporal_weight = temporal_var / (temporal_var + spatial_var + 1e-6)
        spatial_weight = spatial_var / (temporal_var + spatial_var + 1e-6)
        
        # Fuse with adaptive weights
        adaptive_saliency = (temporal_weight * temporal_saliency + 
                           spatial_weight * spatial_saliency)
        
        # Normalize
        adaptive_saliency = adaptive_saliency.astype(np.float32)
        min_val = np.min(adaptive_saliency)
        max_val = np.max(adaptive_saliency)
        
        if max_val - min_val > 1e-6:
            adaptive_saliency = (adaptive_saliency - min_val) / (max_val - min_val)
        else:
            adaptive_saliency = np.zeros_like(adaptive_saliency)
        
        return adaptive_saliency
