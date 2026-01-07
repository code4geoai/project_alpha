# Enhanced Multi-Spectral Prompt Architecture Design

## Executive Summary

This document outlines an incremental enhancement strategy to transform the current robust NDVI prompt system into a comprehensive multi-spectral approach using AI4Boundaries bands (B2, B3, B4, B8). The foundation system is now fully operational with all NDVI variants (temporal, spatial, combined, adaptive) properly bounded to parcels. The enhancement strategy adds harmonic composite prompts as the primary advancement, followed by EVI2-temporal and B2-B3-gradient approaches, maintaining separate prompt types for scientific rigor and risk mitigation.

## Current System Limitations

### 1. **Spectral Information Gap**
- **Current**: Only uses B4 (red) and B8 (NIR) for NDVI computation
- **Missing**: B2 (blue) and B3 (green) spectral information
- **Impact**: Limited discrimination capability for different vegetation types and soil conditions

### 2. **Temporal Analysis Constraints**
- **Current**: Temporal NDVI variance and peak difference from time series (computed in src/step0_ndvi_timeseries.py)
- **Missing**: Multi-spectral temporal composites (MAD metric)
- **Impact**: Unable to capture temporal stability across different spectral bands

### 3. **Spatial Analysis Limitations**
- **Current**: NDVI-only spatial gradients
- **Missing**: Multi-spectral spatial gradients and texture analysis
- **Impact**: Poor boundary detection for spectrally similar crops

### 4. **SAM Integration Basic**
- **Current**: Zero-shot SAM with simple prompts
- **Missing**: Mask decoder fine-tuning for agricultural domain
- **Impact**: Poor adaptation to field boundary characteristics

## How Current NDVI Variants Work
- **Temporal Prompts**: Use NDVI variance + peak difference across time
- **Spatial Prompts**: Compute spatial gradients of NDVI time series
- **Combined Prompts**: Merge temporal + spatial NDVI information
- **Adaptive Prompts**: Automatically select best method (temporal/spatial/combined)

## Incremental Multi-Spectral Enhancement Strategy

### 1. **Current Prompt System (✅ FULLY FUNCTIONAL)**
The existing system includes:
- **Vanilla Prompts**: Geometric-based, not spectral
- **NDVI Temporal Prompts**: Time series analysis using NDVI variance and peak difference across time
- **NDVI Variants** (All Issues Resolved):
  - **Temporal Prompts**: Use NDVI variance + peak difference across time ✅
  - **NDVI-Gradient (Current Spatial)**: Utilize current spatial gradients of NDVI for enhanced spatial analysis ✅
  - **Combined Prompts**: Merge temporal + spatial NDVI information ✅ **(Parcel bounding fixed)**
  - **Adaptive Prompts**: Automatically select best method (temporal/spatial/combined) ✅ **(Parcel bounding fixed)**

#### 1.1 Recent Fixes Applied
- **Parcel Bounding Issue Resolved**: Combined and adaptive prompts now properly contained within parcels
- **Enhanced Border Avoidance**: Added image border buffer to prevent edge artifacts
- **Robust Validation**: Improved parcel containment checking with guaranteed minimum prompts per parcel
- **Consistent Behavior**: All prompt types now use reliable parcel bounding methods

### 2. **Enhanced Multi-Spectral Prompt Types**

#### 2.1 EVI2 Temporal Prompts (Supplementary to NDVI)
- **Formula**: EVI2 = 2.5 × (B8 - B4) / (B8 + 4.4×B4 + 1)
- **Bands**: Uses B8 (NIR) and B4 (Red) - same bands as NDVI but with enhanced coefficients
- **Advantage**: More sensitive to canopy structure, less affected by soil background than NDVI
- **Relationship**: Complements NDVI temporal prompts rather than replacing them
- **Integration**: Extends existing time series processing to compute both NDVI and EVI2 in parallel

#### 2.2 B2-B3 Gradient Prompts (Complementary to NDVI Gradients)
- **Spatial Analysis**: Compute Sobel gradients for B2 (blue) and B3 (green) bands
- **Different from NDVI Gradients**: While existing NDVI spatial prompts use NDVI gradients (gradients of NDVI time series), B2-B3 gradients use direct gradients of blue/green bands
- **Spectral Information**: Detects boundaries invisible to red-NIR vegetation analysis (e.g., soil differences, water stress, early crop stages)
- **Combined Approach**: Can be merged with existing NDVI gradients for enhanced boundary detection
- **Selection**: Prioritize locations with high spectral contrast in blue-green spectrum

### 3. **Fallback Strategy: Harmonic Temporal Composites**
If the incremental approach (NDVI + EVI2 + B2-B3 gradients) fails to achieve desired parcel coverage:
- **Harmonic Regression**: Apply UKFields-style phenological modeling using constant, time, cosine, and sine terms
- **Phase and Amplitude**: Derive temporal stability metrics for enhanced boundary detection
- **Visualization**: Create RGB composites using phase, amplitude, and median values for SAM input

### 4. **Prompt Generation Architecture**

#### 4.1 Separate Prompt Types
Maintain modular design with distinct prompt categories:
1. **Vanilla** (geometric)
2. **NDVI-temporal + variants** (spatial, combined, adaptive)
3. **EVI2-temporal** (enhanced vegetation index)
4. **B2-B3-gradient** (blue-green spectral gradients)
5. **Harmonic-composite** (fallback option)

#### 4.2 Incremental Enhancement Process
- **Phase 1**: Implement EVI2-temporal and validate improvement over NDVI-temporal
- **Phase 2**: Add B2-B3 gradients and measure additional gains
- **Phase 3**: Deploy harmonic composites as fallback if needed
- **Evaluation**: Each phase independently validated before proceeding

## Addressing NDVI Prompt Gaps

The NDVI-based centroid prompts often fail to generate prompts for all parcels in an image due to their reliance solely on red (B4) and near-infrared (B8) bands. This spectral limitation results in poor discrimination for parcels with similar NDVI values but differences in blue (B2) and green (B3) bands, such as certain crop types, soil conditions, or early/late growth stages. Consequently, prompts may cluster in high-NDVI areas, leaving low-contrast or spectrally subtle parcels uncovered.

Research from projects like UKFields demonstrates that harmonic temporal composites, derived from phenological modeling of vegetation indices, significantly improve field boundary detection when used as input for segmentation models like SAM. Building on this, the multi-spectral approach addresses the gap by leveraging all four AI4Boundaries bands (B2, B3, B4, B8) to compute additional vegetation indices like EVI2, which is more sensitive to canopy structure and less affected by soil background. Temporal MAD composites or harmonic regression capture stability and phenological cycles across all spectral bands, reducing noise from clouds or atmosphere that could obscure parcels in NDVI-only analysis. Multi-spectral spatial gradients and texture analysis (e.g., using Sobel filters, LBP, and GLCM) detect boundaries in spectrally similar regions that NDVI gradients miss. Prompt selection algorithms prioritize spectral diversity and spatial distribution within parcels, ensuring broader coverage by identifying high-contrast locations across multiple bands rather than just NDVI peaks. This comprehensive spectral and temporal integration aims to generate prompts for all parcels, filling the gaps inherent in NDVI-based methods and aligning with state-of-the-art practices in agricultural field delineation. The AI4Boundaries dataset's multi-band monthly composites and diverse stratified sampling provide an ideal testbed, ensuring that multi-spectral prompts can be validated against high-accuracy GSAA labels for comprehensive parcel coverage.

## Implementation Architecture

### Layer 1: Enhanced Time Series Processing
- **Purpose**: Generate EVI2 alongside existing NDVI time series
- **Input**: AI4Boundaries Sentinel-2 monthly composites (B2, B3, B4, B8) - 6 months temporal data
- **Output**: NDVI and EVI2 temporal stacks with variance analysis
- **Integration**: Extends `src/step0_ndvi_timeseries.py` to include EVI2

### Layer 2: Multi-Band Spatial Analysis
- **Purpose**: Compute gradients for B2 and B3 bands alongside NDVI
- **Input**: Temporal composites (NDVI, EVI2, B2, B3)
- **Output**: Multi-spectral gradient maps for prompt generation
- **Methods**: Sobel filtering on individual bands with combined saliency

### Layer 3: Modular Prompt Generation
- **Purpose**: Generate separate prompt types with independent validation
- **Input**: Spatial gradients and temporal stability metrics
- **Output**: Five distinct prompt categories
  - Vanilla (geometric)
  - NDVI-temporal variants
  - EVI2-temporal
  - B2-B3-gradient
  - Harmonic-composite (fallback)

### Layer 4: SAM Integration
- **Purpose**: Enhanced SAM prediction using multiple prompt types
- **Input**: Various prompt categories and training data
- **Output**: Improved segmentation with option for fine-tuned decoder

### Layer 5: Evaluation Framework
- **Purpose**: Compare performance across prompt types
- **Input**: Predictions from different prompt categories and ground truth
- **Output**: Performance metrics and comparative analysis
- **Metrics**: Parcel coverage rate, boundary detection accuracy, computational efficiency

## Key Architectural Changes

### 1. **Incremental Prompt Enhancement**
```
Vanilla → NDVI-temporal → EVI2-temporal → B2-B3-gradient → Harmonic-fallback
    ↓           ↓              ↓              ↓                ↓
  SAM       Enhanced      Enhanced      Enhanced        Enhanced
           Parcel      Parcel Coverage  Parcel Coverage  Parcel Coverage
           Coverage        +10%           +15%            +20%
```

### 2. **Modular Integration Strategy**
- **Backward Compatibility**: All existing prompt types remain functional
- **Independent Validation**: Each enhancement measured against previous baseline
- **Scientific Rigor**: Clear attribution of improvements to specific components
- **Risk Management**: Fallback options ensure system reliability

### 3. **Performance Optimization**
- **Memory Efficiency**: EVI2 shares memory footprint with NDVI (same bands)
- **Computational Cost**: B2-B3 gradients add minimal processing overhead
- **Validation Speed**: Separate prompt types enable faster iterative testing
- **Scalability**: Modular design supports adding new spectral indices easily

## Expected Improvements by Phase

### Phase 1: EVI2 Enhancement
- **Current**: NDVI-temporal prompts
- **Enhanced**: EVI2-temporal prompts with improved canopy sensitivity
- **Expected Gain**: 5-15% improvement in parcel coverage, especially for dense vegetation

### Phase 2: B2-B3 Gradient Addition
- **Current**: EVI2-enhanced NDVI system
- **Enhanced**: Multi-band gradient analysis (B2, B3, B4, B8)
- **Expected Gain**: Additional 5-10% coverage improvement for spectrally subtle boundaries

### Phase 3: Harmonic Composite Fallback
- **Current**: Incremental spectral enhancements
- **Enhanced**: Phenological modeling for challenging regions
- **Expected Gain**: Robust performance in edge cases and complex agricultural landscapes

### 4. **Overall System Benefits**
- **Modular Validation**: Clear attribution of improvements to specific components
- **Risk Mitigation**: Fallback mechanisms ensure system reliability
- **Scientific Rigor**: Each enhancement independently validated before integration
- **Scalability**: Foundation for future spectral index additions

## Risk Mitigation - Incremental Approach

### 1. **Performance Validation Risk**
- **Risk**: EVI2 or B2-B3 enhancements may not improve parcel coverage
- **Mitigation**:
  - Independent validation of each enhancement before integration
  - Clear performance metrics (parcel coverage %, boundary accuracy)
  - Fallback to previous working version if enhancement fails

### 2. **Computational Overhead**
- **Risk**: Additional processing requirements for multi-spectral analysis
- **Mitigation**:
  - EVI2 uses same computational footprint as NDVI (same input bands)
  - B2-B3 gradients add minimal processing overhead
  - GPU acceleration for gradient computations when needed

### 3. **System Integration Risk**
- **Risk**: Breaking existing NDVI-based functionality
- **Mitigation**:
  - Maintain backward compatibility with all existing prompt types
  - Modular design allows independent testing and debugging
  - Clear separation between enhancement phases
  - Comprehensive testing on AI4Boundaries dataset before deployment

## Next Steps - Harmonic Composite Implementation

### ✅ Current Status (Parcel Bounding Fixed)
- **All NDVI Variants Operational**: Temporal, spatial, combined, and adaptive prompts now fully functional
- **Parcel Bounding Resolved**: Combined and adaptive prompts properly contained within parcels
- **Robust Validation**: Enhanced parcel containment checking implemented
- **System Stability**: All prompt generation methods working consistently

### 🎯 Next Implementation Phase: Harmonic Composite (Primary Focus)

#### Phase 1: Harmonic Regression Foundation (Week 1-2)
1. **Harmonic Analysis Module**: Create `src/harmonic_analysis.py` for UKFields-style phenological modeling
   - Implement constant, time, cosine, and sine term regression
   - Derive phase and amplitude metrics for temporal stability
   - Add harmonic composite generation for SAM input

2. **Time Series Enhancement**: Extend `src/step0_ndvi_timeseries.py` to support harmonic analysis
   - Integrate harmonic regression with existing NDVI/EVI2 time series
   - Add harmonic stability metrics alongside variance-based approaches
   - Maintain backward compatibility with current prompt generation

3. **Harmonic Prompt Generation**: Create new prompt category in `src/superpixels_temporal.py`
   - Implement `run_harmonic_superpixels()` function
   - Add harmonic-based saliency computation using phase/amplitude metrics
   - Integrate with existing parcel bounding and validation logic

#### Phase 2: Multi-Spectral Integration (Week 3-4)
1. **EVI2 Temporal Enhancement**: Implement enhanced vegetation index alongside NDVI
   - Formula: EVI2 = 2.5 × (B8 - B4) / (B8 + 4.4×B4 + 1)
   - Extend time series processing to compute EVI2 in parallel with NDVI
   - Create EVI2-temporal prompts following existing patterns

2. **B2-B3 Gradient Analysis**: Add blue-green spectral gradient detection
   - Implement Sobel filtering on B2 (blue) and B3 (green) bands
   - Create B2-B3 gradient prompts complementary to NDVI gradients
   - Integrate multi-band spatial analysis with harmonic temporal composites

#### Phase 3: System Integration and Validation (Week 5-6)
1. **Modular Prompt Architecture**: Ensure all prompt types work independently and in combination
   - Vanilla (geometric)
   - NDVI-temporal variants (temporal, spatial, combined, adaptive)
   - EVI2-temporal (enhanced vegetation index)
   - B2-B3-gradient (blue-green spectral gradients)
   - Harmonic-composite (phenological modeling)

2. **Performance Benchmarking**: Comprehensive evaluation against AI4Boundaries ground truth
   - Compare parcel coverage rates across all prompt types
   - Measure boundary detection accuracy improvements
   - Evaluate computational efficiency and memory usage

3. **Fallback Strategy**: Implement intelligent method selection
   - Primary: Harmonic composite for complex agricultural landscapes
   - Secondary: Multi-spectral (EVI2 + B2-B3) for standard cases
   - Tertiary: NDVI variants for simple scenarios
   - Validation: Automatic method selection based on parcel characteristics

### 🚀 Implementation Priorities

1. **Harmonic Composite First**: Focus on phenological modeling as the primary enhancement
2. **Multi-Spectral Enhancement**: Add EVI2 and B2-B3 gradients as complementary approaches
3. **Intelligent Fallback**: Implement smart method selection for optimal parcel coverage
4. **Comprehensive Validation**: Benchmark all approaches on AI4Boundaries dataset

This roadmap transforms the current robust NDVI system into a comprehensive multi-spectral, phenologically-aware prompt generation framework with intelligent fallback mechanisms.