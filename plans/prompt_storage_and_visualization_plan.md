# Spatial Gradient Prompt Storage & Visualization Plan

## 1. Where New Prompts Are Created and Saved

### 1.1 Storage Directory Structure
Based on your existing configuration, new spatial gradient prompts will be saved in:

```python
# Updated config/superpixels_config.py
SPATIAL_PROMPTS_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "spatial_prompts"))
COMBINED_PROMPTS_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "combined_prompts"))

# Update main config.py
from src.config import SPATIAL_PROMPTS_DIR, COMBINED_PROMPTS_DIR
os.makedirs(SPATIAL_PROMPTS_DIR, exist_ok=True)
os.makedirs(COMBINED_PROMPTS_DIR, exist_ok=True)
```

### 1.2 Prompt File Naming Convention
```
# Existing files:
vanilla_prompts_{image_id}.npy        # Geometric centers
superpixel_prompts_{image_id}.npy     # Temporal variance + peak diff

# New spatial gradient files:
spatial_prompts_{image_id}.npy        # Sobel gradients + spatial features
combined_prompts_{image_id}.npy       # Mixed temporal + spatial approach
adaptive_prompts_{image_id}.npy       # Auto-selected best method
```

### 1.3 Prompt Generation Functions
```python
# In src/superpixels_temporal.py - add these new functions

def save_spatial_centroids(image_id, centroids):
    """Save spatial gradient-based prompts"""
    out_path = os.path.join(SPATIAL_PROMPTS_DIR, f"spatial_prompts_{image_id}.npy")
    np.save(out_path, np.array(centroids))
    print(f"[Spatial SPX] Saved {len(centroids)} centroids → {out_path}")
    return out_path

def save_combined_centroids(image_id, centroids):
    """Save combined temporal + spatial prompts"""
    out_path = os.path.join(COMBINED_PROMPTS_DIR, f"combined_prompts_{image_id}.npy")
    np.save(out_path, np.array(centroids))
    print(f"[Combined SPX] Saved {len(centroids)} centroids → {out_path}")
    return out_path

def run_spatial_superpixels(dataset, country="Netherlands"):
    """Generate spatial gradient-based prompts"""
    # Similar to run_temporal_superpixels but uses spatial method
    # Implementation details below...

def run_combined_superpixels(dataset, country="Netherlands"):
    """Generate combined temporal + spatial prompts"""
    # Hybrid approach that uses both methods
```

## 2. Visualization System Extension

### 2.1 Update Existing visualize_prompts.py
Add these new visualization functions to your existing `src/visualize_prompts.py`:

```python
def visualize_spatial_prompts(dataset, country="Netherlands"):
    """Visualize spatial gradient-based prompts"""
    # Load spatial prompts instead of temporal
    spatial_path = os.path.join(SPATIAL_PROMPTS_DIR, f"spatial_prompts_{image_id}.npy")
    # Similar structure to visualize_temporal_prompts but loads spatial prompts

def visualize_vanilla_vs_spatial(dataset, country="Netherlands"):
    """Compare vanilla (centroids) vs spatial gradient prompts"""
    # Two-column plot: vanilla vs spatial
    # Left: vanilla prompts (red circles)
    # Right: spatial prompts (green crosses)

def visualize_temporal_vs_spatial(dataset, country="Netherlands"):
    """Compare temporal vs spatial prompts"""
    # Two-column plot: temporal vs spatial
    # Left: temporal prompts (blue dots) 
    # Right: spatial prompts (green crosses)

def visualize_all_prompt_types(dataset, country="Netherlands"):
    """Show all four prompt types: vanilla, temporal, spatial, combined"""
    # Four subplot comparison:
    # 1. Vanilla (red circles)
    # 2. Temporal (blue dots)
    # 3. Spatial (green crosses)  
    # 4. Combined (magenta triangles)

def visualize_prompt_distribution_analysis(dataset, country="Netherlands"):
    """Analyze spatial distribution quality across methods"""
    # Metrics: clustering coefficient, spatial spread, parcel coverage
    # Bar charts comparing methods
    # Heatmaps showing prompt density
```

### 2.2 Updated Directory References
```python
# In src/visualize_prompts.py - add these imports
from src.config import (
    VANILLA_PROMPTS_DIR,
    TEMPORAL_PROMPTS_DIR,
    SPATIAL_PROMPTS_DIR,      # NEW
    COMBINED_PROMPTS_DIR,     # NEW
)
```

### 2.3 Example Enhanced Visualization Function
```python
def visualize_comprehensive_prompt_comparison(dataset, country="Netherlands"):
    """
    Comprehensive 4-panel comparison:
    - Panel 1: Vanilla prompts (geometric centers)
    - Panel 2: Temporal prompts (variance + peak diff)  
    - Panel 3: Spatial prompts (Sobel gradients)
    - Panel 4: Combined prompts (temporal + spatial)
    
    Also shows:
    - Prompt counts per method
    - Parcel coverage percentages
    - Spatial distribution metrics
    """
    
    # Load all prompt types for each image
    vanilla_path = os.path.join(VANILLA_PROMPTS_DIR, f"vanilla_prompts_{image_id}.npy")
    temporal_path = os.path.join(TEMPORAL_PROMPTS_DIR, f"superpixel_prompts_{image_id}.npy")
    spatial_path = os.path.join(SPATIAL_PROMPTS_DIR, f"spatial_prompts_{image_id}.npy")
    combined_path = os.path.join(COMBINED_PROMPTS_DIR, f"combined_prompts_{image_id}.npy")
    
    # Create 4-subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot each prompt type with different colors/markers
    prompt_configs = [
        (vanilla_path, "red", "o", "Vanilla (Centers)"),
        (temporal_path, "blue", ".", "Temporal (Variance)"),
        (spatial_path, "green", "x", "Spatial (Gradients)"),
        (combined_path, "magenta", "^", "Combined"),
    ]
    
    for idx, (path, color, marker, label) in enumerate(prompt_configs):
        ax = axes[idx//2, idx%2]
        
        # Load prompts
        prompts = np.load(path) if os.path.exists(path) else np.array([])
        
        # Plot RGB background
        ax.imshow(rgb)
        _lock_imshow_limits(ax, rgb)
        
        # Plot parcel boundaries
        _plot_parcel_boundaries_pixels(ax, parcels, item["mask_path"], rgb, 
                                     color="yellow", linewidth=1)
        
        # Plot prompts
        if len(prompts) > 0:
            ax.scatter(prompts[:, 0], prompts[:, 1], c=color, marker=marker, 
                      s=30, alpha=0.8, label=label)
        
        # Calculate and display metrics
        coverage = compute_parcel_coverage(prompts, parcels, x_vals, y_vals)
        ax.set_title(f"{label}\n{len(prompts)} prompts, {coverage:.1f}% coverage")
        ax.axis('off')
    
    plt.suptitle(f"Image {image_id} - Comprehensive Prompt Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()
```

## 3. Integration with Existing Code

### 3.1 Main Entry Points
```python
# Add to src/superpixels_temporal.py
def run_all_superpixel_methods(dataset, country="Netherlands"):
    """Run all prompt generation methods and save results"""
    
    print("🚀 Generating prompts with all methods...\n")
    
    # Method 1: Temporal (existing)
    run_temporal_superpixels(dataset, country)
    
    # Method 2: Spatial (new)
    run_spatial_superpixels(dataset, country)
    
    # Method 3: Combined (new)
    run_combined_superpixels(dataset, country)
    
    print("✅ All prompt generation methods completed!")
```

### 3.2 Updated Configuration
```python
# src/config/superpixels_config.py
SUPERPIXELS_CONFIG = {
    'directories': {
        'temporal': TEMPORAL_PROMPTS_DIR,
        'spatial': SPATIAL_PROMPTS_DIR,
        'combined': COMBINED_PROMPTS_DIR,
        'adaptive': ADAPTIVE_PROMPTS_DIR,  # NEW
    },
    
    'file_patterns': {
        'temporal': 'superpixel_prompts_{image_id}.npy',
        'spatial': 'spatial_prompts_{image_id}.npy',        # NEW
        'combined': 'combined_prompts_{image_id}.npy',      # NEW
        'adaptive': 'adaptive_prompts_{image_id}.npy',      # NEW
    },
    
    'visualization': {
        'colors': {
            'vanilla': 'red',
            'temporal': 'blue', 
            'spatial': 'green',        # NEW
            'combined': 'magenta',     # NEW
        },
        'markers': {
            'vanilla': 'o',
            'temporal': '.',
            'spatial': 'x',            # NEW
            'combined': '^',           # NEW
        }
    }
}
```

## 4. Usage Examples

### 4.1 Generate All Prompt Types
```python
# Generate spatial gradient prompts
from src.superpixels_temporal import run_spatial_superpixels
run_spatial_superpixels(dataset)

# Generate combined prompts  
from src.superpixels_temporal import run_combined_superpixels
run_combined_superpixels(dataset)

# Generate all methods at once
from src.superpixels_temporal import run_all_superpixel_methods
run_all_superpixel_methods(dataset)
```

### 4.2 Visualize Results
```python
# Compare temporal vs spatial
from src.visualize_prompts import visualize_temporal_vs_spatial
visualize_temporal_vs_spatial(dataset)

# Comprehensive 4-way comparison
from src.visualize_prompts import visualize_comprehensive_prompt_comparison
visualize_comprehensive_prompt_comparison(dataset)

# Analyze prompt distribution quality
from src.visualize_prompts import visualize_prompt_distribution_analysis
visualize_prompt_distribution_analysis(dataset)
```

This approach maintains your existing workflow while adding powerful new spatial gradient-based prompt generation and comprehensive visualization tools.