# NDVI Spatial Gradient Integration Plan

## Design Philosophy: Backward-Compatible Modular Architecture

### Core Principles
1. **Zero Breaking Changes**: Existing code continues to work unchanged
2. **Modular Design**: Each NDVI metric in separate, testable module
3. **Plugin Architecture**: Easy to add new metrics without modifying core
4. **Spatially-Aware Selection**: Better prompt distribution across parcels
5. **Progressive Enhancement**: Start with current method, add spatial alternatives

## Proposed File Structure

```
src/
├── superpixels_temporal.py          # Main file (MINIMAL CHANGES)
├── ndvi_metrics/                    # NEW: Modular metrics package
│   ├── __init__.py
│   ├── temporal_metrics.py          # EXISTING: variance, peak_diff
│   ├── spatial_metrics.py           # NEW: gradients, local_variance
│   ├── combined_metrics.py          # NEW: hybrid approaches
│   └── metric_interface.py          # NEW: abstract base class
├── saliency_builders/               # NEW: saliency generation strategies
│   ├── __init__.py
│   ├── temporal_builder.py          # EXISTING: current implementation
│   ├── spatial_builder.py           # NEW: spatial gradient based
│   ├── combined_builder.py          # NEW: mixed temporal/spatial
│   └── adaptive_builder.py          # NEW: automatically selects best method
├── prompt_selection/                # NEW: intelligent prompt selection
│   ├── __init__.py
│   ├── top_k_selector.py            # EXISTING: current top-k approach
│   ├── spatial_diversity_selector.py # NEW: ensures spatial spread
│   ├── boundary_aligned_selector.py  # NEW: follows parcel boundaries
│   └── adaptive_selector.py         # NEW: chooses best strategy per parcel
└── config/
    ├── superpixels_config.py        # NEW: configuration for new methods
```

## Implementation Strategy

### Phase 1: Extract Existing Functionality (Zero Risk)
1. Move temporal metrics to `ndvi_metrics/temporal_metrics.py`
2. Move saliency building to `saliency_builders/temporal_builder.py`
3. Keep `superpixels_temporal.py` as thin wrapper
4. **Result**: Existing functionality works identically

### Phase 2: Add Spatial Capabilities (Low Risk)
1. Implement spatial metrics in `ndvi_metrics/spatial_metrics.py`
2. Create spatial saliency builder
3. Add configuration options
4. **Result**: New spatial method available, old one unchanged

### Phase 3: Enhanced Prompt Selection (Medium Risk)
1. Implement spatial diversity selector
2. Add boundary-aligned selection
3. Make selection strategy configurable
4. **Result**: Better prompt distribution

### Phase 4: Integration & Testing (Low Risk)
1. Add adaptive methods that auto-select best approach
2. Comprehensive testing
3. Performance optimization

## Key Design Decisions

### 1. Backward Compatibility Strategy
```python
# In superpixels_temporal.py - MINIMAL CHANGES
def build_temporal_saliency_mask(ndvi_ts, method='temporal', **kwargs):
    """
    method: 'temporal' (existing) | 'spatial' (new) | 'adaptive' (smart)
    """
    if method == 'temporal':
        return TemporalSaliencyBuilder.build(ndvi_ts, **kwargs)
    elif method == 'spatial':
        return SpatialSaliencyBuilder.build(ndvi_ts, **kwargs)
    elif method == 'adaptive':
        return AdaptiveSaliencyBuilder.build(ndvi_ts, **kwargs)
```

### 2. Plugin Architecture for Metrics
```python
# All metrics implement same interface
class NDVIMetric(ABC):
    @abstractmethod
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        """Returns saliency map [H, W]"""
        pass

# Spatial gradient implementation
class SpatialGradientMetric(NDVIMetric):
    def compute(self, ndvi_ts: np.ndarray) -> np.ndarray:
        # Use median NDVI across time for stability
        stable_ndvi = np.nanmedian(ndvi_ts, axis=0)
        # Apply Sobel filters
        grad_x = ndimage.sobel(stable_ndvi, axis=1)
        grad_y = ndimage.sobel(stable_ndvi, axis=0)
        return np.sqrt(grad_x**2 + grad_y**2)
```

### 3. Configuration System
```python
# config/superpixels_config.py
SUPERPIXELS_CONFIG = {
    'temporal': {
        'var_thresh': 0.05,
        'peak_thresh': 0.15,
        'method': 'variance_peak_diff'
    },
    'spatial': {
        'gradient_thresh': 0.1,
        'local_var_radius': 3,
        'method': 'sobel_gradient'
    },
    'adaptive': {
        'prefer_spatial_for_boundaries': True,
        'temporal_weight': 0.3,
        'spatial_weight': 0.7
    }
}
```

### 4. Enhanced Prompt Selection
```python
def select_diverse_prompts_per_parcel(
    parcels, saliency, x_vals, y_vals, 
    strategy='spatial_diversity', **kwargs
):
    """
    strategy: 'top_k' (existing) | 'spatial_diversity' | 'boundary_aligned'
    """
    if strategy == 'top_k':
        return TopKSelector.select(parcels, saliency, x_vals, y_vals, **kwargs)
    elif strategy == 'spatial_diversity':
        return SpatialDiversitySelector.select(parcels, saliency, x_vals, y_vals, **kwargs)
    elif strategy == 'boundary_aligned':
        return BoundaryAlignedSelector.select(parcels, saliency, x_vals, y_vals, **kwargs)
```

## Migration Path

### Immediate (Zero Risk)
- Current code continues working unchanged
- Add new modules alongside existing code
- Test new methods on small subset

### Short Term (Low Risk)
- Switch to modular architecture for new experiments
- Compare temporal vs spatial methods
- A/B test different prompt selection strategies

### Long Term (Medium Risk)
- Make spatial methods default for agricultural boundary detection
- Deprecate temporal-only approach
- Full migration to adaptive system

## Benefits of This Approach

1. **Risk Mitigation**: Existing functionality never breaks
2. **Incremental Testing**: Test each component independently
3. **Easy Comparison**: Side-by-side comparison of methods
4. **Future-Proof**: Easy to add new metrics and strategies
5. **Maintainable**: Clear separation of concerns
6. **Configurable**: Easy to tune parameters without code changes

## Expected Outcomes

1. **Better Spatial Distribution**: Spatial gradients should distribute prompts more evenly
2. **Improved Boundary Detection**: Focus on actual spatial discontinuities
3. **Maintained Compatibility**: Existing pipelines continue working
4. **Enhanced Flexibility**: Easy to experiment with different approaches
5. **Clear Performance Metrics**: Easy to compare methods quantitatively