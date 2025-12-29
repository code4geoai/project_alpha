# src/eval_gold_standard.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

from segment_anything import sam_model_registry, SamPredictor
from src.lora_sam import load_sam_with_lora
from src.step2_scaling import load_nc_image
from src.metrics import compute_iou, compute_dice, compute_boundary_f1
from src.config import (
    SAM_MODEL_TYPE,
    SAM_CHECKPOINT,
    TEMPORAL_PROMPTS_DIR,
    VANILLA_PROMPTS_DIR,
    SPATIAL_PROMPTS_DIR,
    COMBINED_PROMPTS_DIR,
    ADAPTIVE_PROMPTS_DIR,
)


@torch.no_grad()
def evaluate(
    dataset,
    mode="vanilla",      # "vanilla" | "ndvi" | "ndvi_lora"
    device="cuda",
    lora_ckpt="checkpoints/lora_mask_decoder.pt",
):
    """
    Gold-standard evaluator for SAM ablations.
    """

    assert mode in ["vanilla", "ndvi", "ndvi_lora"]

    # --------------------------------------------------
    # Load SAM
    # --------------------------------------------------
    if mode == "vanilla":
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        predictor = SamPredictor(sam)

    else:
        sam, predictor, _ = load_sam_with_lora(device=device)
        if mode == "ndvi_lora":
            assert os.path.exists(lora_ckpt), "LoRA checkpoint missing"
            lora_state = torch.load(lora_ckpt, map_location=device)
            sam.load_state_dict(lora_state, strict=False)

    sam.eval()

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    ious, dices, boundaries = [], [], []

    for item in dataset:
        image_id = item["id"]
        gt_mask_np = item["mask"]

        if gt_mask_np.sum() == 0:
            continue

        # ---- Load image ----
        _, rgb = load_nc_image(item["nc_path"])
        predictor.set_image((rgb * 255).astype(np.uint8))

        # ---- Load prompts ----
        if mode == "vanilla":
            prompt_path = os.path.join(
                VANILLA_PROMPTS_DIR,
                f"vanilla_prompts_{image_id}.npy"
            )
        else:
            prompt_path = os.path.join(
                TEMPORAL_PROMPTS_DIR,
                f"superpixel_prompts_{image_id}.npy"
            )

        if not os.path.exists(prompt_path):
            continue

        points_np = np.load(prompt_path)
        if len(points_np) == 0:
            continue

        points = torch.tensor(points_np, device=device).float()
        labels = torch.ones(len(points), device=device)

        # ---- Predict (TORCH ONLY) ----
        masks, _, _ = predictor.predict_torch(
            point_coords=points.unsqueeze(0),
            point_labels=labels.unsqueeze(0),
            multimask_output=False,
        )

        pred = (masks[0, 0] > 0.5).float()

        gt = torch.tensor(gt_mask_np, device=device).float()
        gt = F.interpolate(
            gt.unsqueeze(0).unsqueeze(0),
            size=pred.shape,
            mode="nearest",
        )[0, 0]

        # ---- Metrics ----
        ious.append(compute_iou(pred, gt))
        dices.append(compute_dice(pred, gt))
        boundaries.append(compute_boundary_f1(pred, gt))

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------
    results = {
        "iou": float(np.mean(ious)) if len(ious) else np.nan,
        "dice": float(np.mean(dices)) if len(dices) else np.nan,
        "boundary": float(np.mean(boundaries)) if len(boundaries) else np.nan,
    }

    print(f"\n[{mode.upper()}] Mean scores:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


# -------------------------
# COMPREHENSIVE PROMPT EVALUATION
# -------------------------
@torch.no_grad()
def evaluate_all_prompt_types(
    dataset,
    prompt_types=None,  # List of prompt types to evaluate
    device="cuda",
    subset_size=None,  # Evaluate only first N images if specified
    save_plots=True,
    output_dir="evaluation_results"
):
    """
    Comprehensive evaluation of all prompt types with statistical analysis and visualization.
    
    Args:
        dataset: Dataset containing images and masks
        prompt_types: List of prompt types to evaluate ['vanilla', 'temporal', 'spatial', 'combined', 'adaptive']
        device: Device for computation
        subset_size: Evaluate only first N images if specified
        save_plots: Whether to save visualization plots
        output_dir: Directory to save results
    
    Returns:
        dict: Comprehensive evaluation results with statistics and plots
    """
    
    if prompt_types is None:
        prompt_types = ['vanilla', 'temporal', 'spatial', 'combined', 'adaptive']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prompt type to directory mapping
    prompt_dirs = {
        'vanilla': VANILLA_PROMPTS_DIR,
        'temporal': TEMPORAL_PROMPTS_DIR,
        'spatial': SPATIAL_PROMPTS_DIR,
        'combined': COMBINED_PROMPTS_DIR,
        'adaptive': ADAPTIVE_PROMPTS_DIR,
    }
    
    # Prompt type to filename mapping
    prompt_files = {
        'vanilla': lambda image_id: f"vanilla_prompts_{image_id}.npy",
        'temporal': lambda image_id: f"superpixel_prompts_{image_id}.npy",
        'spatial': lambda image_id: f"spatial_prompts_{image_id}.npy",
        'combined': lambda image_id: f"combined_prompts_{image_id}.npy",
        'adaptive': lambda image_id: f"adaptive_prompts_{image_id}.npy",
    }
    
    # Filter prompt types that exist
    available_types = []
    for ptype in prompt_types:
        if ptype in prompt_dirs and os.path.exists(prompt_dirs[ptype]):
            available_types.append(ptype)
    
    if not available_types:
        raise ValueError("No valid prompt types found!")
    
    print(f"🎯 Evaluating {len(available_types)} prompt types: {available_types}")
    
    # Limit dataset size if requested
    eval_dataset = dataset[:subset_size] if subset_size else dataset
    print(f"📊 Evaluating on {len(eval_dataset)} images")
    
    # --------------------------------------------------
    # Load SAM (same model for fair comparison)
    # --------------------------------------------------
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    sam.eval()
    
    # --------------------------------------------------
    # Evaluation storage
    # --------------------------------------------------
    results = {
        'prompt_type': [],
        'image_id': [],
        'iou': [],
        'dice': [],
        'boundary': [],
        'num_prompts': [],
        'success': []
    }
    
    prompt_stats = defaultdict(lambda: {'count': 0, 'total_prompts': 0})
    
    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    for item in eval_dataset:
        image_id = item["id"]
        gt_mask_np = item["mask"]
        
        if gt_mask_np.sum() == 0:
            print(f"⚠️  SKIP: Image {image_id} has empty ground truth mask")
            continue
        
        # Load image once per item
        _, rgb = load_nc_image(item["nc_path"])
        
        # Evaluate each prompt type for this image
        for prompt_type in available_types:
            prompt_path = os.path.join(prompt_dirs[prompt_type], prompt_files[prompt_type](image_id))
            
            if not os.path.exists(prompt_path):
                print(f"⚠️  SKIP: {prompt_type} prompts not found for image {image_id}")
                results['prompt_type'].append(prompt_type)
                results['image_id'].append(image_id)
                results['iou'].append(np.nan)
                results['dice'].append(np.nan)
                results['boundary'].append(np.nan)
                results['num_prompts'].append(0)
                results['success'].append(False)
                continue
            
            try:
                # Load prompts
                points_np = np.load(prompt_path)
                if len(points_np) == 0:
                    print(f"⚠️  SKIP: {prompt_type} prompts empty for image {image_id}")
                    results['prompt_type'].append(prompt_type)
                    results['image_id'].append(image_id)
                    results['iou'].append(np.nan)
                    results['dice'].append(np.nan)
                    results['boundary'].append(np.nan)
                    results['num_prompts'].append(0)
                    results['success'].append(False)
                    continue
                
                # Set image for SAM predictor
                predictor.set_image((rgb * 255).astype(np.uint8))
                
                # Prepare prompts
                points = torch.tensor(points_np, device=device).float()
                labels = torch.ones(len(points), device=device)
                
                # Predict masks
                masks, _, _ = predictor.predict_torch(
                    point_coords=points.unsqueeze(0),
                    point_labels=labels.unsqueeze(0),
                    multimask_output=False,
                )
                
                pred = (masks[0, 0] > 0.5).float()
                
                gt = torch.tensor(gt_mask_np, device=device).float()
                gt = F.interpolate(
                    gt.unsqueeze(0).unsqueeze(0),
                    size=pred.shape,
                    mode="nearest",
                )[0, 0]
                
                # Compute metrics
                iou = compute_iou(pred, gt)
                dice = compute_dice(pred, gt)
                boundary = compute_boundary_f1(pred, gt)
                
                # Store results
                results['prompt_type'].append(prompt_type)
                results['image_id'].append(image_id)
                results['iou'].append(iou)
                results['dice'].append(dice)
                results['boundary'].append(boundary)
                results['num_prompts'].append(len(points_np))
                results['success'].append(True)
                
                # Update stats
                prompt_stats[prompt_type]['count'] += 1
                prompt_stats[prompt_type]['total_prompts'] += len(points_np)
                
                print(f"✅ {prompt_type:>8} | Image {image_id} | IoU: {iou:.3f} | Dice: {dice:.3f} | Boundary: {boundary:.3f}")
                
            except Exception as e:
                print(f"❌ ERROR: {prompt_type} on image {image_id}: {str(e)}")
                results['prompt_type'].append(prompt_type)
                results['image_id'].append(image_id)
                results['iou'].append(np.nan)
                results['dice'].append(np.nan)
                results['boundary'].append(np.nan)
                results['num_prompts'].append(0)
                results['success'].append(False)
    
    # --------------------------------------------------
    # Statistical Analysis
    # --------------------------------------------------
    df = pd.DataFrame(results)
    
    # Compute summary statistics
    summary_stats = []
    for prompt_type in available_types:
        type_data = df[(df['prompt_type'] == prompt_type) & (df['success'] == True)]
        
        if len(type_data) > 0:
            stats = {
                'Prompt_Type': prompt_type,
                'Images_Evaluated': len(type_data),
                'Total_Prompts': prompt_stats[prompt_type]['total_prompts'],
                'Avg_Prompts_per_Image': prompt_stats[prompt_type]['total_prompts'] / len(type_data),
                'IoU_Mean': type_data['iou'].mean(),
                'IoU_Std': type_data['iou'].std(),
                'Dice_Mean': type_data['dice'].mean(),
                'Dice_Std': type_data['dice'].std(),
                'Boundary_Mean': type_data['boundary'].mean(),
                'Boundary_Std': type_data['boundary'].std(),
            }
        else:
            stats = {
                'Prompt_Type': prompt_type,
                'Images_Evaluated': 0,
                'Total_Prompts': 0,
                'Avg_Prompts_per_Image': 0,
                'IoU_Mean': np.nan,
                'IoU_Std': np.nan,
                'Dice_Mean': np.nan,
                'Dice_Std': np.nan,
                'Boundary_Mean': np.nan,
                'Boundary_Std': np.nan,
            }
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    
    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    if save_plots:
        create_evaluation_plots(df, summary_df, output_dir)
    
    # --------------------------------------------------
    # Print Summary
    # --------------------------------------------------
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE PROMPT EVALUATION SUMMARY")
    print("="*80)
    
    print("\n📊 SUMMARY STATISTICS:")
    print(summary_df.round(4).to_string(index=False))
    
    print("\n📈 PROMPT AVAILABILITY:")
    for prompt_type in available_types:
        type_data = df[df['prompt_type'] == prompt_type]
        success_count = type_data['success'].sum()
        total_count = len(type_data)
        availability = (success_count / total_count * 100) if total_count > 0 else 0
        print(f"  {prompt_type:>8}: {success_count:>3}/{total_count:<3} images ({availability:>5.1f}% available)")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("="*80)
    
    return {
        'detailed_results': df,
        'summary_statistics': summary_df,
        'prompt_stats': dict(prompt_stats),
        'output_directory': output_dir
    }


def create_evaluation_plots(df, summary_df, output_dir):
    """
    Create comprehensive evaluation plots.
    """
    
    # Filter successful evaluations only
    success_data = df[df['success'] == True].copy()
    
    if len(success_data) == 0:
        print("⚠️  No successful evaluations to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    
    # 1. Performance comparison box plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['iou', 'dice', 'boundary']
    metric_names = ['IoU', 'Dice', 'Boundary F1']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        data_by_type = []
        labels = []
        
        for prompt_type in summary_df['Prompt_Type']:
            type_data = success_data[success_data['prompt_type'] == prompt_type][metric]
            if len(type_data) > 0:
                data_by_type.append(type_data.values)
                labels.append(prompt_type)
        
        if data_by_type:
            axes[i].boxplot(data_by_type, labels=labels)
            axes[i].set_title(f'{name} Distribution by Prompt Type')
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean performance comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(summary_df))
    width = 0.25
    
    ax.bar(x - width, summary_df['IoU_Mean'], width, label='IoU', alpha=0.8)
    ax.bar(x, summary_df['Dice_Mean'], width, label='Dice', alpha=0.8)
    ax.bar(x + width, summary_df['Boundary_Mean'], width, label='Boundary F1', alpha=0.8)
    
    ax.set_xlabel('Prompt Type')
    ax.set_ylabel('Score')
    ax.set_title('Mean Performance Comparison Across Prompt Types')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Prompt_Type'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Prompt availability and count analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prompt availability
    availability_data = []
    for prompt_type in summary_df['Prompt_Type']:
        type_data = df[df['prompt_type'] == prompt_type]
        success_count = type_data['success'].sum()
        total_count = len(type_data)
        availability = (success_count / total_count * 100) if total_count > 0 else 0
        availability_data.append(availability)
    
    ax1.bar(summary_df['Prompt_Type'], availability_data, alpha=0.8, color='skyblue')
    ax1.set_xlabel('Prompt Type')
    ax1.set_ylabel('Availability (%)')
    ax1.set_title('Prompt Type Availability Across Dataset')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Average prompts per image
    ax2.bar(summary_df['Prompt_Type'], summary_df['Avg_Prompts_per_Image'], alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Prompt Type')
    ax2.set_ylabel('Average Prompts per Image')
    ax2.set_title('Average Prompt Count by Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Evaluation plots saved to: {output_dir}")


# -------------------------
# CONVENIENCE FUNCTIONS
# -------------------------
def evaluate_prompt_types_quick(dataset, prompt_types=None, subset_size=10):
    """
    Quick evaluation on a small subset with basic results.
    """
    return evaluate_all_prompt_types(
        dataset=dataset,
        prompt_types=prompt_types,
        subset_size=subset_size,
        save_plots=True,
        output_dir="quick_evaluation_results"
    )


def evaluate_spatial_vs_temporal(dataset, subset_size=None):
    """
    Specific comparison between spatial and temporal prompts.
    """
    return evaluate_all_prompt_types(
        dataset=dataset,
        prompt_types=['spatial', 'temporal'],
        subset_size=subset_size,
        save_plots=True,
        output_dir="spatial_vs_temporal_evaluation"
    )


def evaluate_all_methods_comprehensive(dataset, subset_size=None):
    """
    Comprehensive evaluation of all available prompt methods.
    """
    return evaluate_all_prompt_types(
        dataset=dataset,
        prompt_types=['vanilla', 'temporal', 'spatial', 'combined', 'adaptive'],
        subset_size=subset_size,
        save_plots=True,
        output_dir="comprehensive_evaluation_results"
    )
