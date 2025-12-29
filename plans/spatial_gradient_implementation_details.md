# NDVI Spatial Gradient Integration - Implementation Guide

This document provides the implementation approach for integrating spatial gradient-based NDVI metrics while maintaining full backward compatibility with the existing superpixels_temporal.py codebase.

## Phase 1: Zero-Risk Foundation (Extract & Modularize)

### Step 1: Create Modular Package Structure

Create the following directory structure:

```
src/ndvi_metrics/
├── __init__.py
├── temporal_metrics.py          # Existing variance + peak_diff
├── spatial_metrics.py           # NEW: Sobel gradients + spatial features
├── combined_metrics.py          # NEW: hybrid temporal/spatial
└── metric_interface.py          # NEW: abstract base class

src/saliency_builders/
├── __init__.py
├── temporal_builder.py          # EXISTING: current saliency building
├── spatial_builder.py           # NEW: spatial gradient saliency
├── combined_builder.py          # NEW: mixed approach
└── adaptive_builder.py          # NEW: auto-select best method

src/prompt_selection/
├── __init__.py
├── top_k_selector.py            # EXISTING: current top-k approach
├── spatial_diversity_selector.py # NEW: ensures spatial spread
├── boundary_aligned_selector.py  # NEW: follows parcel boundaries
└── adaptive_selector.py         # NEW: chooses strategy per parcel

src/config/
└── superpixels_config.py        # NEW: configuration for new methods
```

## Step 2: Implementation Details

### 2.1 Metric Interface
Abstract base class for all NDVI-based saliency metrics. All metrics implement compute() method returning [H, W] saliency map.

### 2.2 Temporal Metrics
- NDVI temporal variance metric (existing implementation)
- NDVI peak difference metric (existing implementation)
- Combined temporal variance + peak difference (existing approach)

### 2.3 Spatial Metrics
- Sobel gradient metric: Primary metric for agricultural boundary detection using spatial gradients
- Local variance metric: Areas with high local variance are likely boundaries
- Quantile-based metric: Highlights consistently high/low NDVI areas
- Combined spatial metric: Weighted combination for robust boundary detection

## Integration with Existing Code

### 3.1 Backward-Compatible Wrapper
Minimal changes to superpixels_temporal.py to support multiple saliency methods ('temporal', 'spatial', 'adaptive', 'combined').

### 3.2 Enhanced Prompt Selection
Support for multiple selection strategies ('top_k', 'spatial_diversity', 'boundary_aligned').

## Usage Examples

### 4.1 Current Usage (Backward Compatible)
Existing code continues to work unchanged.

### 4.2 New Spatial Method
Use spatial gradients for boundary detection with spatial diversity selection.

### 4.3 Adaptive Method
Let system choose best method automatically based on preferences.

## Configuration System

### 5.1 Easy Parameter Tuning
Configurable parameters for different use cases (boundary detection, temporal robustness, adaptive).

## Benefits Summary

1. **Zero Breaking Changes**: All existing code continues to work
2. **Modular Design**: Easy to test and modify individual components
3. **Spatial Focus**: Primary metric targets agricultural boundaries
4. **Configurable**: Easy parameter tuning without code changes
5. **Future-Proof**: Plugin architecture for easy extension
6. **Progressive Enhancement**: Start with current method, add spatial alternatives

This implementation provides a robust foundation for integrating spatial gradient-based NDVI metrics while maintaining full backward compatibility with the existing codebase.