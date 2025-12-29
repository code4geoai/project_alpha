# Comprehensive Prompt Evaluation System Guide

## Overview

The enhanced evaluation system in `src/eval_gold_standard.py` now supports comprehensive evaluation of all prompt types with statistical analysis and visualization capabilities.

## New Features

### ✅ **Complete Prompt Type Support**
- **Vanilla**: Traditional centroid-based prompts
- **Temporal**: NDVI time-series analysis prompts  
- **Spatial**: Spatial gradient-based prompts
- **Combined**: Fusion of temporal + spatial signals
- **Adaptive**: Auto-selects best method per image

### ✅ **Comprehensive Analysis**
- **Statistical Summary**: Mean, std, availability for all metrics
- **Detailed Results**: Per-image performance breakdown
- **Visual Analysis**: Box plots, comparison charts, availability analysis
- **Graceful Handling**: Manages missing/empty prompt files

### ✅ **Fair Evaluation Design**
- **Same SAM Model**: All prompt types evaluated with identical SAM configuration
- **Natural Selection Awareness**: Accounts for different selection criteria per prompt type
- **Subset Support**: Test on small samples before full evaluation

## Key Functions

### 1. `evaluate_all_prompt_types()` - Main Function

```python
from src.eval_gold_standard import evaluate_all_prompt_types

results = evaluate_all_prompt_types(
    dataset=your_dataset,
    prompt_types=['vanilla', 'temporal', 'spatial', 'combined', 'adaptive'],
    device='cuda',
    subset_size=10,  # Start small for testing
    save_plots=True,
    output_dir='evaluation_results'
)
```

**Returns:**
- `detailed_results`: DataFrame with per-image results
- `summary_statistics`: Aggregated statistics by prompt type  
- `prompt_stats`: Prompt availability and count info
- `output_directory`: Path to saved results

### 2. `evaluate_prompt_types_quick()` - Quick Test

```python
from src.eval_gold_standard import evaluate_prompt_types_quick

# Quick evaluation on 10 images
results = evaluate_prompt_types_quick(
    dataset=your_dataset,
    subset_size=10
)
```

### 3. `evaluate_spatial_vs_temporal()` - Specific Comparison

```python
from src.eval_gold_standard import evaluate_spatial_vs_temporal

# Compare spatial vs temporal prompts
results = evaluate_spatial_vs_temporal(
    dataset=your_dataset,
    subset_size=15
)
```

### 4. `evaluate_all_methods_comprehensive()` - Full Analysis

```python
from src.eval_gold_standard import evaluate_all_methods_comprehensive

# Complete evaluation of all methods
results = evaluate_all_methods_comprehensive(
    dataset=your_dataset,
    subset_size=20
)
```

## Output Files

### 📊 **Statistical Results**
- **`detailed_results.csv`**: Per-image performance metrics
- **`summary_statistics.csv`**: Aggregated statistics by prompt type

### 📈 **Visualizations**
- **`performance_comparison_boxplots.png`**: IoU/Dice/Boundary distributions
- **`mean_performance_comparison.png`**: Average performance comparison
- **`prompt_analysis.png`**: Availability and count analysis

## Metrics Evaluated

### **IoU (Intersection over Union)**
```python
# Measures overlap between predicted and ground truth masks
# Range: [0, 1], where 1 = perfect overlap
```

### **Dice Coefficient**
```python
# Measures similarity between predicted and ground truth
# Range: [0, 1], where 1 = perfect match
```

### **Boundary F1**
```python
# Measures boundary detection accuracy
# Lower values = better boundary alignment
```

## Handling Missing Data

The system gracefully handles:

### **Missing Prompt Files**
- Skips images where prompt files don't exist
- Records as unsuccessful evaluation
- Continues with other prompt types

### **Empty Prompt Arrays**
- Detects when prompt files exist but contain no valid prompts
- Skips evaluation for that image-type combination
- Common for NDVI-based prompts in unsuitable areas

### **Ground Truth Issues**
- Skips images with empty or invalid ground truth masks
- Ensures meaningful evaluation only

## Example Usage Workflow

### **Step 1: Quick Test**
```python
# Test evaluation system on small subset
results = evaluate_prompt_types_quick(dataset, subset_size=5)
```

### **Step 2: Method Comparison**
```python
# Compare specific methods of interest
results = evaluate_spatial_vs_temporal(dataset, subset_size=15)
```

### **Step 3: Comprehensive Analysis**
```python
# Full evaluation when ready
results = evaluate_all_prompt_types(
    dataset, 
    prompt_types=['vanilla', 'temporal', 'spatial', 'combined', 'adaptive'],
    subset_size=None,  # Full dataset
    output_dir='final_evaluation_results'
)
```

## Expected Output Summary

```
🎯 COMPREHENSIVE PROMPT EVALUATION SUMMARY
================================================================================

📊 SUMMARY STATISTICS:
Prompt_Type  Images_Evaluated  Total_Prompts  Avg_Prompts_per_Image  IoU_Mean  ...
vanilla                   25             325                   13.0    0.654  ...
temporal                  18             156                    8.7    0.723  ...
spatial                   20             203                   10.2    0.698  ...
combined                  12              89                    7.4    0.745  ...
adaptive                  15             134                    8.9    0.731  ...

📈 PROMPT AVAILABILITY:
   vanilla:  25/25 images (100.0% available)
  temporal:  18/25 images ( 72.0% available)
   spatial:  20/25 images ( 80.0% available)
   combined:  12/25 images ( 48.0% available)
   adaptive:  15/25 images ( 60.0% available)

💾 Results saved to: evaluation_results
```

## Key Insights from Results

### **Performance Hierarchy**
- Compare mean IoU/Dice scores across prompt types
- Higher scores indicate better segmentation performance

### **Availability Patterns**
- Vanilla prompts: High availability (universal coverage)
- NDVI-based prompts: Selective availability (quality filtering)
- Combined prompts: Lowest availability (highest selectivity)

### **Efficiency Analysis**
- Average prompts per image: Balance between coverage and computational cost
- Performance vs. selectivity trade-offs

## Troubleshooting

### **Common Issues**

**1. CUDA Memory Errors**
```python
# Reduce batch size or use smaller subset
subset_size=5  # Start smaller
```

**2. Missing Prompt Files**
```python
# Check that prompt generation completed successfully
# Verify file paths in config.py
```

**3. SAM Model Errors**
```python
# Ensure SAM checkpoint is available
# Check SAM_MODEL_TYPE and SAM_CHECKPOINT in config.py
```

**4. Empty Ground Truth**
```python
# Some images may have no valid agricultural masks
# System will skip these automatically
```

### **Performance Optimization**

**1. Start Small**
```python
# Always test with subset_size=5-10 first
# Scale up after verifying everything works
```

**2. Monitor Memory**
```python
# Large datasets may require smaller batches
# Consider evaluating in chunks for full datasets
```

**3. Use Appropriate Subset**
```python
# Representative sample for quick testing
# Full dataset for final results
```

## Best Practices

1. **Always start with small subsets** for testing
2. **Verify prompt files exist** before full evaluation
3. **Check ground truth quality** for your dataset
4. **Use consistent SAM configuration** for fair comparison
5. **Save results** with descriptive output directories
6. **Analyze availability patterns** to understand natural selection

## Next Steps

After evaluation:

1. **Analyze statistical significance** of performance differences
2. **Investigate failure cases** for underperforming methods
3. **Consider hybrid approaches** combining best-performing methods
4. **Plan LoRA training** using best-performing prompt strategies
5. **Iterate on prompt generation** based on evaluation insights