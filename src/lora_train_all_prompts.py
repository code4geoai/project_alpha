# src/lora_train_all_prompts.py
# Helper functions for training LoRA with all prompt types

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from src.lora_train import train_lora
from src.config import (
    TEMPORAL_PROMPTS_DIR,
    EVI2_PROMPTS_DIR,
    B2B3_PROMPTS_DIR,
    SPATIAL_PROMPTS_DIR,
    COMBINED_PROMPTS_DIR,
    ADAPTIVE_PROMPTS_DIR,
    VANILLA_PROMPTS_DIR,
)


def train_all_prompt_types_individual(
    dataset,
    prompt_types: List[str] = None,
    base_ckpt_path: str = "checkpoints/lora_mask_decoder",
    **train_kwargs
) -> Dict[str, torch.nn.Module]:
    """
    Train separate LoRA models for each prompt type.
    
    Args:
        dataset: Training dataset
        prompt_types: List of prompt types to train. If None, uses all available.
        base_ckpt_path: Base checkpoint path (will be suffixed with prompt type)
        **train_kwargs: Additional arguments passed to train_lora
    
    Returns:
        Dictionary mapping prompt_type to trained model
    """
    if prompt_types is None:
        prompt_types = ["temporal", "evi2", "b2b3", "spatial", "combined", "adaptive", "vanilla"]
    
    trained_models = {}
    
    for prompt_type in prompt_types:
        print(f"\n{'='*60}")
        print(f"Training LoRA with {prompt_type.upper()} prompts")
        print(f"{'='*60}")
        
        # Create prompt-specific checkpoint path
        ckpt_path = f"{base_ckpt_path}_{prompt_type}.pt"
        
        # Filter out non-train_lora parameters
        filtered_kwargs = {k: v for k, v in train_kwargs.items() 
                          if k not in ['min_images']}
        
        # Train model for this prompt type
        model = train_lora(
            dataset=dataset,
            temporal_prompt_dir=TEMPORAL_PROMPTS_DIR,
            prompt_type=prompt_type,
            ckpt_path=ckpt_path,
            **filtered_kwargs
        )
        
        trained_models[prompt_type] = model
        
    return trained_models


def train_mixed_prompts(
    dataset,
    prompt_types: List[str] = None,
    max_points_per_type: int = 8,
    epochs_per_type: int = 5,
    **train_kwargs
):
    """
    Train a single LoRA model using mixed prompts from all types.
    Cycles through prompt types during training.
    
    Args:
        dataset: Training dataset
        prompt_types: List of prompt types to use. If None, uses all available.
        max_points_per_type: Maximum points to use from each prompt type
        epochs_per_type: Number of epochs to train on each prompt type before switching
        **train_kwargs: Additional arguments passed to train_lora
    """
    if prompt_types is None:
        prompt_types = ["temporal", "evi2", "b2b3", "spatial", "combined", "adaptive", "vanilla"]
    
    print(f"\n{'='*60}")
    print(f"Training LoRA with MIXED prompts: {prompt_types}")
    print(f"{'='*60}")
    
    # Import here to avoid circular imports
    from src.lora_sam import load_sam_with_lora
    from src.step2_scaling import load_nc_image
    from src.lora_train import compute_iou
    
    # Setup
    device = train_kwargs.get("device", "cuda")
    ckpt_path = train_kwargs.get("ckpt_path", "checkpoints/lora_mask_decoder_mixed.pt")
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    # Prompt type mappings
    prompt_type_to_dir = {
        'temporal': TEMPORAL_PROMPTS_DIR,
        'evi2': EVI2_PROMPTS_DIR,
        'b2b3': B2B3_PROMPTS_DIR,
        'spatial': SPATIAL_PROMPTS_DIR,
        'combined': COMBINED_PROMPTS_DIR,
        'adaptive': ADAPTIVE_PROMPTS_DIR,
        'vanilla': VANILLA_PROMPTS_DIR,
    }
    
    prompt_type_to_filename = {
        'temporal': lambda image_id: f"superpixel_prompts_{image_id}.npy",
        'evi2': lambda image_id: f"evi2_prompts_{image_id}.npy",
        'b2b3': lambda image_id: f"b2b3_prompts_{image_id}.npy",
        'spatial': lambda image_id: f"spatial_prompts_{image_id}.npy",
        'combined': lambda image_id: f"combined_prompts_{image_id}.npy",
        'adaptive': lambda image_id: f"adaptive_prompts_{image_id}.npy",
        'vanilla': lambda image_id: f"vanilla_prompts_{image_id}.npy",
    }
    
    # Load model
    sam, predictor, lora_params = load_sam_with_lora(device=device)
    optimizer = torch.optim.AdamW(lora_params, lr=train_kwargs.get("lr", 1e-4))
    
    sam.train()
    sam.mask_decoder.train()
    
    total_epochs = len(prompt_types) * epochs_per_type
    current_epoch = 0
    
    # Training loop
    for type_idx, prompt_type in enumerate(prompt_types):
        print(f"\n🎯 Training with {prompt_type.upper()} prompts (Round {type_idx + 1}/{len(prompt_types)})")
        
        prompt_dir = prompt_type_to_dir[prompt_type]
        prompt_filename = prompt_type_to_filename[prompt_type]
        
        for epoch in range(epochs_per_type):
            current_epoch += 1
            print(f"\n[Mixed Training] Epoch {current_epoch}/{total_epochs} (Type: {prompt_type})")
            
            epoch_loss = 0.0
            epoch_iou = 0.0
            n_samples = 0
            
            for i, item in enumerate(dataset):
                image_id = item["id"]
                gt_mask = item["mask"]
                
                if gt_mask.sum() == 0:
                    continue
                
                # Load prompts for this type
                prompt_file = os.path.join(prompt_dir, prompt_filename(image_id))
                if not os.path.exists(prompt_file):
                    continue
                
                try:
                    points_np = np.load(prompt_file)[:max_points_per_type]
                except:
                    continue
                
                if len(points_np) == 0:
                    continue
                
                print(f"  → Image {image_id} ({i+1}/{len(dataset)}) - {prompt_type}")
                
                # Standard training steps
                _, rgb = load_nc_image(item["nc_path"])
                predictor.set_image((rgb * 255).astype(np.uint8))
                
                points = torch.tensor(points_np, device=device, dtype=torch.float32)
                labels = torch.ones(len(points), device=device)
                
                # Pad to max_points if necessary
                max_points = train_kwargs.get("max_points", 32)
                n = len(points)
                if n < max_points:
                    padding_points = torch.zeros(max_points - n, 2, device=device)
                    padding_labels = torch.zeros(max_points - n, device=device)
                    points = torch.cat([points, padding_points], dim=0)
                    labels = torch.cat([labels, padding_labels], dim=0)
                
                # Forward pass
                image_embeddings = predictor.get_image_embedding()
                
                with torch.no_grad():
                    sparse_pe, dense_pe = sam.prompt_encoder(
                        points=(points.unsqueeze(0), labels.unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )
                    image_pe = sam.prompt_encoder.get_dense_pe()
                
                low_res_logits, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_pe,
                    dense_prompt_embeddings=dense_pe,
                    multimask_output=False,
                )
                
                pred_logits = low_res_logits[0, 0]
                pred_prob = torch.sigmoid(pred_logits)
                
                gt = torch.tensor(gt_mask, device=device).float()
                gt_low = torch.nn.functional.interpolate(
                    gt.unsqueeze(0).unsqueeze(0),
                    size=pred_logits.shape[-2:],
                    mode="nearest",
                )[0, 0]
                
                # Loss and optimization
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, gt_low)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                iou = compute_iou(pred_prob, gt_low)
                
                epoch_loss += loss.item()
                epoch_iou += iou.item()
                n_samples += 1
            
            mean_loss = epoch_loss / max(n_samples, 1)
            mean_iou = epoch_iou / max(n_samples, 1)
            
            print(f"[Epoch {current_epoch}] Mean loss = {mean_loss:.4f} | Mean IoU = {mean_iou:.4f}")
            
            # Save checkpoint
            if current_epoch % train_kwargs.get("checkpoint_interval", 5) == 0:
                ckpt_file = ckpt_path.replace('.pt', f'_epoch_{current_epoch}.pt')
                torch.save({
                    'epoch': current_epoch,
                    'model_state': {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
                    'optimizer_state': optimizer.state_dict(),
                }, ckpt_file)
                print(f"💾 Checkpoint saved at epoch {current_epoch}")
    
    # Save final model
    if train_kwargs.get("save_final", True):
        final_path = ckpt_path.replace('.pt', '_final.pt')
        torch.save(
            {k: v.cpu() for k, v in sam.state_dict().items() if "lora_" in k},
            final_path
        )
        print(f"💾 Final mixed prompts model saved to {final_path}")
    
    return sam


def check_prompt_availability(dataset, prompt_types: List[str] = None) -> Dict[str, int]:
    """
    Check how many images have prompts available for each prompt type.
    
    Args:
        dataset: Dataset to check
        prompt_types: List of prompt types to check. If None, checks all available.
    
    Returns:
        Dictionary mapping prompt_type to count of available prompts
    """
    if prompt_types is None:
        prompt_types = ["temporal", "evi2", "b2b3", "spatial", "combined", "adaptive", "vanilla"]
    
    prompt_type_to_dir = {
        'temporal': TEMPORAL_PROMPTS_DIR,
        'evi2': EVI2_PROMPTS_DIR,
        'b2b3': B2B3_PROMPTS_DIR,
        'spatial': SPATIAL_PROMPTS_DIR,
        'combined': COMBINED_PROMPTS_DIR,
        'adaptive': ADAPTIVE_PROMPTS_DIR,
        'vanilla': VANILLA_PROMPTS_DIR,
    }
    
    prompt_type_to_filename = {
        'temporal': lambda image_id: f"superpixel_prompts_{image_id}.npy",
        'evi2': lambda image_id: f"evi2_prompts_{image_id}.npy",
        'b2b3': lambda image_id: f"b2b3_prompts_{image_id}.npy",
        'spatial': lambda image_id: f"spatial_prompts_{image_id}.npy",
        'combined': lambda image_id: f"combined_prompts_{image_id}.npy",
        'adaptive': lambda image_id: f"adaptive_prompts_{image_id}.npy",
        'vanilla': lambda image_id: f"vanilla_prompts_{image_id}.npy",
    }
    
    availability = {ptype: 0 for ptype in prompt_types}
    
    print("Checking prompt availability...")
    for item in dataset:
        image_id = item["id"]
        for prompt_type in prompt_types:
            prompt_dir = prompt_type_to_dir[prompt_type]
            prompt_filename = prompt_type_to_filename[prompt_type]
            prompt_file = os.path.join(prompt_dir, prompt_filename(image_id))
            
            if os.path.exists(prompt_file):
                availability[prompt_type] += 1
    
    print("\nPrompt Availability Summary:")
    print("-" * 40)
    for prompt_type, count in availability.items():
        percentage = (count / len(dataset)) * 100
        print(f"{prompt_type:12}: {count:3d}/{len(dataset)} images ({percentage:5.1f}%)")
    
    return availability


# Convenience function for easy usage
def quick_train_all_prompts(dataset, strategy="individual", **kwargs):
    """
    Quick training function with all prompts using specified strategy.
    
    Args:
        dataset: Training dataset
        strategy: "individual" or "mixed"
            - "individual": Train separate models for each prompt type
            - "mixed": Train one model with mixed prompts
        **kwargs: Additional training arguments
    """
    # Check availability first
    availability = check_prompt_availability(dataset)
    
    # Filter to only prompt types with sufficient data
    min_images = kwargs.get("min_images", 10)
    available_types = [ptype for ptype, count in availability.items() if count >= min_images]
    
    if not available_types:
        print(f"No prompt types have at least {min_images} images available!")
        return None
    
    print(f"\nUsing prompt types with >= {min_images} images: {available_types}")
    
    # Filter out non-train_lora parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['min_images', 'max_points_per_type', 'epochs_per_type']}
    
    if strategy == "individual":
        return train_all_prompt_types_individual(dataset, available_types, **filtered_kwargs)
    elif strategy == "mixed":
        return train_mixed_prompts(dataset, available_types, **filtered_kwargs)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'individual' or 'mixed'.")