# src/ndvi_metrics/temporal_metrics.py

import numpy as np
from .metric_interface import NormalizedMetric

class NDVIVarianceMetric(NormalizedMetric):
    """NDVI temporal variance metric (extracted from existing implementation)"""
    
    @property
    def name(self) -> str:
        return "ndvi_variance"
    
    @property
    def description(self) -> str:
        return "Temporal variance of NDVI across time series"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """ndvi_ts: NDVI time-series [T, H, W] -> variance map [H, W]"""
        var_map = np.var(ndvi_ts.astype(np.float32), axis=0)
        return self.normalize(var_map)

class NDVIPeakDifferenceMetric(NormalizedMetric):
    """NDVI peak difference metric (extracted from existing implementation)"""
    
    @property
    def name(self) -> str:
        return "ndvi_peak_diff"
    
    @property
    def description(self) -> str:
        return "Difference between max and min NDVI values"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """ndvi_ts: NDVI time-series [T, H, W] -> peak difference map [H, W]"""
        ndvi_max = ndvi_ts.max(axis=0)
        ndvi_min = ndvi_ts.min(axis=0)
        peak_map = (ndvi_max - ndvi_min).astype(np.float32)
        return self.normalize(peak_map)

class CombinedTemporalMetric(NormalizedMetric):
    """Combined temporal variance + peak difference (existing approach)"""

    def __init__(self, var_weight=0.5, peak_weight=0.5):
        self.var_weight = var_weight
        self.peak_weight = peak_weight
        self.variance_metric = NDVIVarianceMetric()
        self.peak_metric = NDVIPeakDifferenceMetric()

    @property
    def name(self) -> str:
        return "combined_temporal"

    @property
    def description(self) -> str:
        return f"Combined temporal metrics (var={self.var_weight}, peak={self.peak_weight})"

    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        var_map = self.variance_metric.compute(ndvi_ts)
        peak_map = self.peak_metric.compute(ndvi_ts)

        combined = (self.var_weight * var_map +
                   self.peak_weight * peak_map)
        return self.normalize(combined)


class EVIVarianceMetric(NormalizedMetric):
    """EVI2 temporal variance metric"""

    @property
    def name(self) -> str:
        return "evi2_variance"

    @property
    def description(self) -> str:
        return "Temporal variance of EVI2 across time series"

    def compute(self, evi2_ts: np.ndarray) -> np.ndarray:
        """evi2_ts: EVI2 time-series [T, H, W] -> variance map [H, W]"""
        var_map = np.var(evi2_ts.astype(np.float32), axis=0)
        return self.normalize(var_map)


class EVIPeakDifferenceMetric(NormalizedMetric):
    """EVI2 peak difference metric"""

    @property
    def name(self) -> str:
        return "evi2_peak_diff"

    @property
    def description(self) -> str:
        return "Difference between max and min EVI2 values"

    def compute(self, evi2_ts: np.ndarray) -> np.ndarray:
        """evi2_ts: EVI2 time-series [T, H, W] -> peak difference map [H, W]"""
        evi2_max = evi2_ts.max(axis=0)
        evi2_min = evi2_ts.min(axis=0)
        peak_map = (evi2_max - evi2_min).astype(np.float32)
        return self.normalize(peak_map)


class CombinedEVITemporalMetric(NormalizedMetric):
    """Combined EVI2 temporal variance + peak difference"""

    def __init__(self, var_weight=0.5, peak_weight=0.5):
        self.var_weight = var_weight
        self.peak_weight = peak_weight
        self.variance_metric = EVIVarianceMetric()
        self.peak_metric = EVIPeakDifferenceMetric()

    @property
    def name(self) -> str:
        return "combined_evi_temporal"

    @property
    def description(self) -> str:
        return f"Combined EVI2 temporal metrics (var={self.var_weight}, peak={self.peak_weight})"

    def compute(self, evi2_ts: np.ndarray) -> np.ndarray:
        var_map = self.variance_metric.compute(evi2_ts)
        peak_map = self.peak_metric.compute(evi2_ts)

        combined = (self.var_weight * var_map +
                   self.peak_weight * peak_map)
        return self.normalize(combined)
