# src/saliency_builders/adaptive_builder.py
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from .temporal_builder import TemporalSaliencyBuilder
from .spatial_builder import SpatialSaliencyBuilder
from .combined_builder import CombinedSaliencyBuilder

class AdaptiveSaliencyBuilder:
    """
    Adaptive saliency builder that automatically selects the best approach.
    """
    
    @staticmethod
    def build(ndvi_ts: np.ndarray,
              preference='boundary_detection',
              use_cuda=True) -> tuple:
        """
        Build saliency mask using adaptive method selection.
        
        Args:
            ndvi_ts: NDVI time-series [T, H, W]
            preference: 'boundary_detection' | 'temporal_stability' | 'adaptive'
            use_cuda: whether to use CUDA acceleration
            
        Returns:
            saliency_mask: float32 [H, W] in [0, 1]
            method_info: dict with method selection info
        """
        if preference == 'boundary_detection':
            return AdaptiveSaliencyBuilder._boundary_detection_method(ndvi_ts, use_cuda)
        elif preference == 'temporal_stability':
            return AdaptiveSaliencyBuilder._temporal_stability_method(ndvi_ts, use_cuda)
        elif preference == 'adaptive':
            return AdaptiveSaliencyBuilder._fully_adaptive_method(ndvi_ts, use_cuda)
        else:
            raise ValueError(f"Unknown preference: {preference}")
    
    @staticmethod
    def _boundary_detection_method(ndvi_ts: np.ndarray, use_cuda=True):
        """
        Primary method for agricultural boundary detection.
        Uses spatial gradients as they directly detect spatial discontinuities.
        """
        spatial_saliency = SpatialSaliencyBuilder.build(ndvi_ts, use_cuda=use_cuda)
        
        method_info = {
            'selected_method': 'spatial_gradients',
            'reason': 'Primary method for agricultural boundary detection',
            'confidence': 0.9
        }
        
        return spatial_saliency, method_info
    
    @staticmethod
    def _temporal_stability_method(ndvi_ts: np.ndarray, use_cuda=True):
        """
        Method that prioritizes temporal consistency.
        Useful for regions with high temporal noise.
        """
        temporal_saliency = TemporalSaliencyBuilder.build(ndvi_ts)
        
        method_info = {
            'selected_method': 'temporal_variance',
            'reason': 'Prioritizes temporal stability over spatial boundaries',
            'confidence': 0.7
        }
        
        return temporal_saliency, method_info
    
    @staticmethod
    def _fully_adaptive_method(ndvi_ts: np.ndarray, use_cuda=True):
        """
        Fully adaptive method that evaluates multiple approaches and selects the best.
        """
        # Compute all saliency maps
        temporal_saliency = TemporalSaliencyBuilder.build(ndvi_ts)
        spatial_saliency = SpatialSaliencyBuilder.build(ndvi_ts, use_cuda=use_cuda)
        combined_saliency = CombinedSaliencyBuilder.build(ndvi_ts, use_cuda=use_cuda)
        
        # Evaluate each method
        temporal_score = AdaptiveSaliencyBuilder._evaluate_saliency(temporal_saliency)
        spatial_score = AdaptiveSaliencyBuilder._evaluate_saliency(spatial_saliency)
        combined_score = AdaptiveSaliencyBuilder._evaluate_saliency(combined_saliency)
        
        # Select best method
        scores = {
            'temporal': temporal_score,
            'spatial': spatial_score, 
            'combined': combined_score
        }
        
        best_method = max(scores, key=scores.get)
        best_saliency = {
            'temporal': temporal_saliency,
            'spatial': spatial_saliency,
            'combined': combined_saliency
        }[best_method]
        
        method_info = {
            'selected_method': best_method,
            'reason': f'Adaptive selection based on saliency quality scores',
            'confidence': scores[best_method],
            'all_scores': scores
        }
        
        return best_saliency, method_info
    
    @staticmethod
    def _evaluate_saliency(saliency: np.ndarray) -> float:
        """
        Evaluate saliency quality using multiple criteria.
        
        Args:
            saliency: saliency map [H, W]
            
        Returns:
            quality_score: float quality score
        """
        # Criterion 1: Dynamic range (higher is better)
        dynamic_range = np.max(saliency) - np.min(saliency)
        
        # Criterion 2: Spatial coherence (lower variance in local regions is better)
        from scipy import ndimage
        local_variance = ndimage.generic_filter(
            saliency, np.var, size=5
        )
        coherence_score = 1.0 / (1.0 + np.mean(local_variance))
        
        # Criterion 3: Edge density (moderate density is optimal)
        from scipy.ndimage import binary_dilation
        thresholded = saliency > np.percentile(saliency, 75)
        edge_density = np.sum(binary_dilation(thresholded)) / saliency.size
        edge_score = 1.0 / (1.0 + abs(edge_density - 0.3))  # Optimal around 30%
        
        # Combined score
        quality_score = (0.4 * dynamic_range + 
                        0.3 * coherence_score + 
                        0.3 * edge_score)
        
        return quality_score
    
    @staticmethod
    def auto_select_for_dataset(dataset_stats: dict, use_cuda=True):
        """
        Auto-select the best method for an entire dataset based on statistics.
        
        Args:
            dataset_stats: dictionary with dataset characteristics
            use_cuda: whether to use CUDA acceleration
            
        Returns:
            recommended_method: str recommended method name
            method_config: dict with recommended configuration
        """
        # Analyze dataset characteristics
        temporal_variability = dataset_stats.get('temporal_variability', 0.5)
        spatial_heterogeneity = dataset_stats.get('spatial_heterogeneity', 0.5)
        noise_level = dataset_stats.get('noise_level', 0.3)
        
        # Decision logic
        if spatial_heterogeneity > 0.7 and noise_level < 0.5:
            # High spatial variation, low noise - ideal for spatial gradients
            recommended_method = 'spatial'
            confidence = 0.9
        elif temporal_variability > 0.7:
            # High temporal variation - use combined approach
            recommended_method = 'combined'
            confidence = 0.8
        elif noise_level > 0.6:
            # High noise - use temporal approach for stability
            recommended_method = 'temporal'
            confidence = 0.7
        else:
            # Default to spatial for agricultural boundaries
            recommended_method = 'spatial'
            confidence = 0.6
        
        method_config = {
            'method': recommended_method,
            'use_cuda': use_cuda,
            'confidence': confidence,
            'reasoning': f"Selected based on dataset characteristics: "
                         f"spatial_het={spatial_heterogeneity:.2f}, "
                         f"temporal_var={temporal_variability:.2f}, "
                         f"noise={noise_level:.2f}"
        }
        
        return recommended_method, method_config
