# src/enhanced_evaluation_metrics.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import xarray as xr
from scipy import ndimage
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


def compute_spectral_diversity_score(prompts, spectral_stacks):
    """
    Compute spectral diversity score for multi-spectral prompts.
    Higher scores indicate better spectral coverage and diversity.
    
    Args:
        prompts: (N, 2) array of prompt coordinates [x, y]
        spectral_stacks: dict of spectral band stacks
        
    Returns:
        float: spectral diversity score [0, 1]
    """
    if len(prompts) == 0:
        return 0.0
    
    diversity_scores = []
    
    for band_name, stack in spectral_stacks.items():
        # Get spectral values at prompt locations
        prompt_values = []
        for x, y in prompts:
            x, y = int(x), int(y)
            if 0 <= x < stack.shape[2] and 0 <= y < stack.shape[1]:
                # Use median across time for stability
                value = np.nanmedian(stack[:, y, x])
                if not np.isnan(value):
                    prompt_values.append(value)
        
        if len(prompt_values) > 1:
            # Compute variance as diversity measure
            variance = np.var(prompt_values)
            # Normalize by band-specific range (rough approximation)
            normalized_variance = variance / (np.nanmax(stack) - np.nanmin(stack) + 1e-6)
            diversity_scores.append(normalized_variance)
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def compute_spectral_contrast_score(prompts, mad_composites):
    """
    Compute spectral contrast score for multi-spectral prompts.
    Measures how well prompts capture spectral boundaries.
    
    Args:
        prompts: (N, 2) array of prompt coordinates [x, y]
        mad_composites: dict of MAD composite arrays
        
    Returns:
        float: spectral contrast score [0, 1]
    """
    if len(prompts) == 0:
        return 0.0
    
    contrast_scores = []
    
    for composite_name, composite_data in mad_composites.items():
        # Get MAD values at prompt locations
        prompt_values = []
        for x, y in prompts:
            x, y = int(x), int(y)
            if 0 <= x < composite_data.shape[1] and 0 <= y < composite_data.shape[0]:
                value = composite_data[y, x]
                if not np.isnan(value):
                    prompt_values.append(value)
        
        if len(prompt_values) > 1:
            # Compute gradient magnitude at prompts
            # Higher MAD values indicate more temporal variation (better boundaries)
            mean_mad = np.mean(prompt_values)
            # Normalize by composite range
            normalized_mad = (mean_mad - np.nanmin(composite_data)) / (np.nanmax(composite_data) - np.nanmin(composite_data) + 1e-6)
            contrast_scores.append(normalized_mad)
    
    return np.mean(contrast_scores) if contrast_scores else 0.0


def compute_spectral_uniformity_score(prompts, gradient_maps):
    """
    Compute spectral uniformity score for multi-spectral prompts.
    Measures consistency across different spectral bands.
    
    Args:
        prompts: (N, 2) array of prompt coordinates [x, y]
        gradient_maps: dict of gradient magnitude maps
        
    Returns:
        float: spectral uniformity score [0, 1]
    """
    if len(prompts) == 0:
        return 0.0
    
    # Get gradient values at prompt locations for each band
    band_gradients = {}
    for grad_name, grad_map in gradient_maps.items():
        band_name = grad_name.replace('_MAD_grad', '')
        prompt_gradients = []
        
        for x, y in prompts:
            x, y = int(x), int(y)
            if 0 <= x < grad_map.shape[1] and 0 <= y < grad_map.shape[0]:
                value = grad_map[y, x]
                if not np.isnan(value):
                    prompt_gradients.append(value)
        
        if prompt_gradients:
            band_gradients[band_name] = prompt_gradients
    
    if len(band_gradients) < 2:
        return 0.0
    
    # Compute uniformity as inverse of coefficient of variation across bands
    band_means = [np.mean(grads) for grads in band_gradients.values()]
    
    if np.mean(band_means) == 0:
        return 0.0
    
    # Coefficient of variation across bands
    cv = np.std(band_means) / (np.mean(band_means) + 1e-6)
    
    # Convert to uniformity score (lower CV = higher uniformity)
    uniformity_score = 1.0 / (1.0 + cv)
    
    return uniformity_score


def compute_multispectral_prompt_metrics(prompts, spectral_stacks, mad_composites, gradient_maps):
    """
    Compute comprehensive multi-spectral prompt evaluation metrics.
    
    Args:
        prompts: (N, 2) array of prompt coordinates [x, y]
        spectral_stacks: dict of spectral band stacks
        mad_composites: dict of MAD composite arrays
        gradient_maps: dict of gradient magnitude maps
        
    Returns:
        dict: comprehensive multi-spectral metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['num_prompts'] = len(prompts)
    
    if len(prompts) == 0:
        metrics.update({
            'spectral_diversity': 0.0,
            'spectral_contrast': 0.0,
            'spectral_uniformity': 0.0,
            'overall_score': 0.0
        })
        return metrics
    
    # Multi-spectral metrics
    metrics['spectral_diversity'] = compute_spectral_diversity_score(prompts, spectral_stacks)
    metrics['spectral_contrast'] = compute_spectral_contrast_score(prompts, mad_composites)
    metrics['spectral_uniformity'] = compute_spectral_uniformity_score(prompts, gradient_maps)
    
    # Overall score (weighted combination)
    weights = {
        'spectral_diversity': 0.4,
        'spectral_contrast': 0.4,
        'spectral_uniformity': 0.2
    }
    
    overall_score = sum(weights[key] * metrics[key] for key in weights.keys())
    metrics['overall_score'] = overall_score
    
    return metrics


def compare_prompt_types_spectral_quality(prompts_dict, spectral_data_dict):
    """
    Compare spectral quality across different prompt types.
    
    Args:
        prompts_dict: dict of {prompt_type: prompts_array}
        spectral_data_dict: dict containing spectral_stacks, mad_composites, gradient_maps
        
    Returns:
        dict: comparison results
    """
    results = {}
    
    for prompt_type, prompts in prompts_dict.items():
        if prompts is not None and len(prompts) > 0:
            metrics = compute_multispectral_prompt_metrics(
                prompts,
                spectral_data_dict['spectral_stacks'],
                spectral_data_dict['mad_composites'],
                spectral_data_dict['gradient_maps']
            )
            results[prompt_type] = metrics
        else:
            results[prompt_type] = {
                'num_prompts': 0,
                'spectral_diversity': 0.0,
                'spectral_contrast': 0.0,
                'spectral_uniformity': 0.0,
                'overall_score': 0.0
            }
    
    return results


def enhanced_evaluation_with_spectral_metrics(dataset, prompt_types=None, subset_size=None):
    """
    Enhanced evaluation that includes spectral quality metrics for multi-spectral prompts.
    
    Args:
        dataset: evaluation dataset
        prompt_types: list of prompt types to evaluate
        subset_size: limit to subset for testing
        
    Returns:
        dict: enhanced evaluation results
    """
    from src.eval_gold_standard import evaluate_all_prompt_types
    from src.generate_multispectral_prompts import compute_multispectral_stack, compute_mad_temporal_composites, compute_spectral_gradients
    
    # Get basic evaluation results
    basic_results = evaluate_all_prompt_types(
        dataset=dataset,
        prompt_types=prompt_types,
        subset_size=subset_size,
        save_plots=False,  # We'll handle plotting separately
        output_dir="temp_eval"
    )
    
    # Add spectral quality analysis for multi-spectral prompts
    if 'multispectral' in basic_results['summary_statistics']['Prompt_Type'].values:
        print("🔬 Adding spectral quality analysis for multi-spectral prompts...")
        
        # For a subset of images, compute spectral quality metrics
        eval_dataset = dataset[:min(10, len(dataset))] if subset_size else dataset[:10]
        
        spectral_quality_results = []
        
        for item in eval_dataset:
            image_id = item["id"]
            
            try:
                # Load spectral data
                spectral_stacks = compute_multispectral_stack(item["nc_path"])
                mad_composites = compute_mad_temporal_composites(spectral_stacks)
                gradient_maps = compute_spectral_gradients(mad_composites)
                
                spectral_data = {
                    'spectral_stacks': spectral_stacks,
                    'mad_composites': mad_composites,
                    'gradient_maps': gradient_maps
                }
                
                # Load prompts for each type
                prompts_dict = {}
                for prompt_type in prompt_types or ['multispectral']:
                    try:
                        from src.config import MULTISPECTRAL_PROMPTS_DIR, VANILLA_PROMPTS_DIR, TEMPORAL_PROMPTS_DIR, SPATIAL_PROMPTS_DIR, COMBINED_PROMPTS_DIR, ADAPTIVE_PROMPTS_DIR, EVI2_PROMPTS_DIR, B2B3_PROMPTS_DIR
                        
                        prompt_dirs = {
                            'multispectral': MULTISPECTRAL_PROMPTS_DIR,
                            'vanilla': VANILLA_PROMPTS_DIR,
                            'temporal': TEMPORAL_PROMPTS_DIR,
                            'spatial': SPATIAL_PROMPTS_DIR,
                            'combined': COMBINED_PROMPTS_DIR,
                            'adaptive': ADAPTIVE_PROMPTS_DIR,
                            'evi2': EVI2_PROMPTS_DIR,
                            'b2b3': B2B3_PROMPTS_DIR,
                        }
                        
                        if prompt_type in prompt_dirs:
                            prompt_path = os.path.join(prompt_dirs[prompt_type], f"{prompt_type}_prompts_{image_id}.npy")
                            if os.path.exists(prompt_path):
                                prompts_dict[prompt_type] = np.load(prompt_path)
                            else:
                                prompts_dict[prompt_type] = None
                        
                    except Exception as e:
                        print(f"Warning: Could not load {prompt_type} prompts for image {image_id}: {e}")
                        prompts_dict[prompt_type] = None
                
                # Compare spectral quality
                quality_comparison = compare_prompt_types_spectral_quality(prompts_dict, spectral_data)
                quality_comparison['image_id'] = image_id
                
                spectral_quality_results.append(quality_comparison)
                
            except Exception as e:
                print(f"Warning: Could not compute spectral quality for image {image_id}: {e}")
                continue
        
        # Add spectral quality results to basic results
        basic_results['spectral_quality_analysis'] = spectral_quality_results
        
        # Print summary
        print("\n🔬 SPECTRAL QUALITY ANALYSIS SUMMARY:")
        print("="*60)
        
        for prompt_type in prompt_types or ['multispectral']:
            if prompt_type in spectral_quality_results[0]:
                type_scores = [r[prompt_type]['overall_score'] for r in spectral_quality_results if prompt_type in r]
                if type_scores:
                    mean_score = np.mean(type_scores)
                    print(f"{prompt_type:>12}: Overall Score = {mean_score:.4f}")
    
    return basic_results


if __name__ == "__main__":
    # Example usage
    from src.step2_scaling import load_dataset
    
    dataset = load_dataset()
    
    results = enhanced_evaluation_with_spectral_metrics(
        dataset=dataset,
        prompt_types=['multispectral', 'temporal'],
        subset_size=5
    )
    
    print("Enhanced evaluation completed!")