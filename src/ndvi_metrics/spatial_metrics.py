# src/ndvi_metrics/spatial_metrics.py

import numpy as np
from .metric_interface import NormalizedMetric

# Optional scipy imports
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None

# Optional PyTorch imports for CUDA acceleration
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

class SobelGradientMetric(NormalizedMetric):
    """
    NDVI spatial gradient using Sobel filters with CUDA acceleration.
    This is the PRIMARY metric for agricultural boundary detection.
    """
    
    def __init__(self, use_median=True, gradient_threshold=0.1, use_cuda=True):
        self.use_median = use_median
        self.gradient_threshold = gradient_threshold
        self.use_cuda = use_cuda and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu') if TORCH_AVAILABLE else None
    
    @property
    def name(self) -> str:
        return "sobel_gradient"
    
    @property
    def description(self) -> str:
        return "Spatial gradient of NDVI using Sobel filters with CUDA acceleration"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradients from NDVI time series.
        
        Args:
            ndvi_ts: NDVI time series [T, H, W]
            
        Returns:
            gradient_magnitude: [H, W] gradient magnitude map
        """
        if self.use_cuda and TORCH_AVAILABLE:
            return self._compute_cuda(ndvi_ts)
        else:
            if self.use_cuda and not TORCH_AVAILABLE:
                print("Warning: PyTorch not available, falling back to CPU computation")
            return self._compute_cpu(ndvi_ts)
    
    def _compute_cpu(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """CPU-based gradient computation using scipy or manual implementation"""
        if not SCIPY_AVAILABLE:
            return self._compute_manual_gradient(ndvi_ts)
            
        # Use median NDVI across time for stability (removes temporal noise)
        if self.use_median:
            stable_ndvi = np.nanmedian(ndvi_ts, axis=0)
        else:
            stable_ndvi = np.nanmean(ndvi_ts, axis=0)
        
        # Apply Sobel filters to compute spatial gradients
        grad_x = ndimage.sobel(stable_ndvi, axis=1)
        grad_y = ndimage.sobel(stable_ndvi, axis=0)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply threshold to focus on strong boundaries
        if self.gradient_threshold > 0:
            gradient_magnitude = np.maximum(
                gradient_magnitude - self.gradient_threshold, 0
            )
        
        return self.normalize(gradient_magnitude)
    
    def _compute_manual_gradient(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Manual gradient computation when scipy is not available"""
        # Use median NDVI across time for stability
        if self.use_median:
            stable_ndvi = np.nanmedian(ndvi_ts, axis=0)
        else:
            stable_ndvi = np.nanmean(ndvi_ts, axis=0)
        
        H, W = stable_ndvi.shape
        grad_x = np.zeros_like(stable_ndvi)
        grad_y = np.zeros_like(stable_ndvi)
        
        # Manual Sobel operator implementation
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Apply convolution manually
        for i in range(1, H-1):
            for j in range(1, W-1):
                # Gradient X
                grad_x[i, j] = (
                    -stable_ndvi[i-1, j-1] - 2*stable_ndvi[i, j-1] - stable_ndvi[i+1, j-1] +
                    stable_ndvi[i-1, j+1] + 2*stable_ndvi[i, j+1] + stable_ndvi[i+1, j+1]
                )
                
                # Gradient Y
                grad_y[i, j] = (
                    -stable_ndvi[i-1, j-1] - 2*stable_ndvi[i-1, j] - stable_ndvi[i-1, j+1] +
                    stable_ndvi[i+1, j-1] + 2*stable_ndvi[i+1, j] + stable_ndvi[i+1, j+1]
                )
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply threshold
        if self.gradient_threshold > 0:
            gradient_magnitude = np.maximum(
                gradient_magnitude - self.gradient_threshold, 0
            )
        
        return self.normalize(gradient_magnitude)
    
    def _compute_cuda(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """CUDA-accelerated gradient computation"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for CUDA computation")
            
        # Convert to torch tensor and move to GPU
        ndvi_tensor = torch.from_numpy(ndvi_ts).float().to(self.device)
        
        # Use median NDVI across time for stability
        if self.use_median:
            stable_ndvi = torch.nanmedian(ndvi_tensor, dim=0)[0]
        else:
            stable_ndvi = torch.nanmean(ndvi_tensor, dim=0)
        
        # Add batch and channel dimensions for conv2d
        stable_ndvi = stable_ndvi.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Define Sobel kernels
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2], 
                                  [-1, 0, 1]]]], dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]]]], dtype=torch.float32, device=self.device)
        
        # Apply convolution
        grad_x = F.conv2d(stable_ndvi, sobel_x, padding=1)
        grad_y = F.conv2d(stable_ndvi, sobel_y, padding=1)
        
        # Gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Remove batch and channel dimensions
        gradient_magnitude = gradient_magnitude.squeeze(0).squeeze(0)
        
        # Apply threshold to focus on strong boundaries
        if self.gradient_threshold > 0:
            gradient_magnitude = torch.clamp(
                gradient_magnitude - self.gradient_threshold, min=0
            )
        
        # Convert back to numpy and normalize
        gradient_np = gradient_magnitude.cpu().numpy()
        return self.normalize(gradient_np)

class LocalVarianceMetric(NormalizedMetric):
    """
    Local spatial variance within NDVI time series.
    Areas with high local variance are likely boundaries.
    """
    
    def __init__(self, window_size=3):
        self.window_size = window_size
    
    @property
    def name(self) -> str:
        return "local_variance"
    
    @property
    def description(self) -> str:
        return "Local spatial variance of NDVI"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Compute local spatial variance"""
        if not SCIPY_AVAILABLE:
            return self._compute_manual_variance(ndvi_ts)
            
        # Use median NDVI across time for stability
        stable_ndvi = np.nanmedian(ndvi_ts, axis=0)
        
        # Compute local variance using convolution
        kernel = np.ones((self.window_size, self.window_size))
        kernel = kernel / kernel.size
        
        # Local mean
        local_mean = ndimage.convolve(stable_ndvi, kernel, mode='constant')
        
        # Local variance
        local_sq_mean = ndimage.convolve(stable_ndvi**2, kernel, mode='constant')
        local_variance = local_sq_mean - local_mean**2
        
        return self.normalize(np.sqrt(np.maximum(local_variance, 0)))
    
    def _compute_manual_variance(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Manual local variance computation when scipy is not available"""
        stable_ndvi = np.nanmedian(ndvi_ts, axis=0)
        H, W = stable_ndvi.shape
        local_var = np.zeros_like(stable_ndvi)
        
        half_window = self.window_size // 2
        
        for i in range(H):
            for j in range(W):
                # Define window bounds
                i_start = max(0, i - half_window)
                i_end = min(H, i + half_window + 1)
                j_start = max(0, j - half_window)
                j_end = min(W, j + half_window + 1)
                
                # Extract window
                window = stable_ndvi[i_start:i_end, j_start:j_end]
                
                # Compute local variance
                local_var[i, j] = np.var(window)
        
        return self.normalize(np.sqrt(np.maximum(local_var, 0)))

class NDVIQuantileMetric(NormalizedMetric):
    """
    Quantile-based NDVI features for spatial diversity.
    Highlights areas with consistently high/low NDVI values.
    """
    
    def __init__(self, quantile_type='high_low'):
        self.quantile_type = quantile_type
    
    @property
    def name(self) -> str:
        return f"ndvi_quantile_{self.quantile_type}"
    
    @property
    def description(self) -> str:
        return f"NDVI {self.quantile_type} quantile features"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Compute quantile-based features"""
        # Compute quantiles across time
        ndvi_q10 = np.nanpercentile(ndvi_ts, 10, axis=0)
        ndvi_q50 = np.nanpercentile(ndvi_ts, 50, axis=0)
        ndvi_q90 = np.nanpercentile(ndvi_ts, 90, axis=0)
        
        if self.quantile_type == 'high':
            # Areas consistently high in NDVI (healthy vegetation)
            return self.normalize(ndvi_q90)
        elif self.quantile_type == 'low':
            # Areas consistently low in NDVI (bare soil, infrastructure)
            return self.normalize(-ndvi_q10)
        elif self.quantile_type == 'high_low':
            # Combine high and low NDVI areas for spatial diversity
            high_areas = self.normalize(ndvi_q90)
            low_areas = self.normalize(-ndvi_q10)
            return high_areas + low_areas
        else:
            return self.normalize(ndvi_q90)

class CombinedSpatialMetric(NormalizedMetric):
    """
    Combined spatial metrics for robust boundary detection.
    """
    
    def __init__(self, 
                 gradient_weight=0.6, 
                 variance_weight=0.2, 
                 quantile_weight=0.2,
                 use_cuda=True):
        self.gradient_weight = gradient_weight
        self.variance_weight = variance_weight
        self.quantile_weight = quantile_weight
        self.use_cuda = use_cuda
        
        self.gradient_metric = SobelGradientMetric(use_cuda=use_cuda)
        self.variance_metric = LocalVarianceMetric()
        self.quantile_metric = NDVIQuantileMetric()
    
    @property
    def name(self) -> str:
        return "combined_spatial"
    
    @property
    def description(self) -> str:
        return "Combined spatial gradient metrics for agricultural boundary detection"
    
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Compute combined spatial saliency"""
        grad_map = self.gradient_metric.compute(ndvi_ts)
        var_map = self.variance_metric.compute(ndvi_ts)
        quant_map = self.quantile_metric.compute(ndvi_ts)
        
        combined = (self.gradient_weight * grad_map + 
                   self.variance_weight * var_map + 
                   self.quantile_weight * quant_map)
        
        return self.normalize(combined)
