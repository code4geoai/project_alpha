# src/ndvi_metrics/metric_interface.py

from abc import ABC, abstractmethod
import numpy as np

class NDVIMetric(ABC):
    """
    Abstract base class for all NDVI-based saliency metrics.
    All metrics must implement compute() method returning [H, W] saliency map.
    """
    
    @abstractmethod
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """
        Compute saliency map from NDVI time series.
        
        Args:
            ndvi_ts: NDVI time series [T, H, W]
            
        Returns:
            saliency_map: float32 array [H, W] in range [0, 1]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name for identification"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return metric description"""
        pass

class NormalizedMetric(NDVIMetric):
    """
    Base class for metrics that need normalization to [0, 1] range.
    """
    
    def normalize(self, saliency: np.ndarray) -> np.ndarray:
        """Normalize saliency map to [0, 1] range"""
        saliency = saliency.astype(np.float32)
        min_val = np.nanmin(saliency)
        max_val = np.nanmax(saliency)
        
        if max_val - min_val < 1e-6:
            return np.zeros_like(saliency)
        
        return (saliency - min_val) / (max_val - min_val)
