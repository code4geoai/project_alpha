# Usage Examples for Training with All Prompts
# ============================================

# Example 1: Train Individual Models for Each Prompt Type
# -------------------------------------------------------
from src.lora_train_all_prompts import train_all_prompt_types_individual
from src.config import TEMPORAL_PROMPTS_DIR

# Train separate LoRA models for each prompt type
models = train_all_prompt_types_individual(
    dataset=dataset,
    prompt_types=["temporal", "evi2", "b2b3", "spatial"],  # Specify which types to use
    epochs=20,
    lr=1e-4,
    device="cuda",
    save_final=True,
    checkpoint_interval=5,
    resume_ckpt=None
)

# This will create separate checkpoints:
# - checkpoints/lora_mask_decoder_temporal.pt
# - checkpoints/lora_mask_decoder_evi2.pt
# - checkpoints/lora_mask_decoder_b2b3.pt
# - checkpoints/lora_mask_decoder_spatial.pt


# Example 2: Quick Training with All Available Prompts
# ---------------------------------------------------
from src.lora_train_all_prompts import quick_train_all_prompts

# Automatically train with all prompt types that have sufficient data
model = quick_train_all_prompts(
    dataset=dataset,
    strategy="individual",  # or "mixed"
    epochs=20,
    lr=1e-4,
    device="cuda",
    min_images=10  # Only use prompt types with at least 10 images
)


# Example 3: Mixed Prompts Training (Single Model)
# ------------------------------------------------
from src.lora_train_all_prompts import train_mixed_prompts

# Train one model that uses prompts from multiple types
model = train_mixed_prompts(
    dataset=dataset,
    prompt_types=["temporal", "evi2", "b2b3"],
    max_points_per_type=8,  # Use 8 points from each type
    epochs_per_type=5,      # 5 epochs per prompt type
    lr=1e-4,
    device="cuda",
    save_final=True,
    checkpoint_interval=5
)


# Example 4: Check Prompt Availability
# -----------------------------------
from src.lora_train_all_prompts import check_prompt_availability

# Check how many images have prompts for each type
availability = check_prompt_availability(dataset)

# Output example:
# Prompt Availability Summary:
# ----------------------------------------
# temporal     : 85/100 images ( 85.0%)
# evi2         : 76/100 images ( 76.0%)
# b2b3         : 76/100 images ( 76.0%)
# spatial      : 76/100 images ( 76.0%)
# combined     : 76/100 images ( 76.0%)
# adaptive     : 76/100 images ( 76.0%)
# vanilla      : 76/100 images ( 76.0%)


# Example 5: Custom Training with Specific Prompt Types
# ----------------------------------------------------
from src.lora_train_all_prompts import train_all_prompt_types_individual

# Train only the most effective prompt types based on your evaluation
models = train_all_prompt_types_individual(
    dataset=dataset,
    prompt_types=["evi2", "b2b3"],  # Focus on EVI2 and B2B3
    epochs=25,
    lr=5e-5,  # Lower learning rate for fine-tuning
    device="cuda",
    save_final=True,
    checkpoint_interval=3,  # More frequent checkpoints
    resume_ckpt=None
)


# Example 6: Resume Training for Specific Prompt Type
# --------------------------------------------------
from src.lora_train import train_lora
from src.config import TEMPORAL_PROMPTS_DIR

# Resume training for a specific prompt type
model = train_lora(
    dataset=dataset,
    temporal_prompt_dir=TEMPORAL_PROMPTS_DIR,
    prompt_type="evi2",
    epochs=30,  # Continue for more epochs
    lr=1e-4,
    device="cuda",
    save_final=True,
    checkpoint_interval=5,
    resume_ckpt="checkpoints/lora_mask_decoder_evi2_ckpt_epoch_20.pt"  # Resume from epoch 20
)


# Strategy Recommendations:
# ========================

# 1. **Individual Models Strategy** (Recommended for comparison):
#    - Pros: Can compare performance of each prompt type
#    - Cons: Requires more storage and training time
#    - Use when: You want to understand which prompt types work best

# 2. **Mixed Prompts Strategy** (Recommended for best performance):
#    - Pros: Combines strengths of all prompt types, potentially better performance
#    - Cons: More complex training, harder to debug
#    - Use when: You want the best possible performance regardless of individual prompt type effectiveness

# 3. **Quick Training Strategy** (Recommended for exploration):
#    - Pros: Easy to set up, automatically handles missing data
#    - Cons: Less control over individual aspects
#    - Use when: You want to quickly test multiple approaches

# Tips:
# =====
# - Always check prompt availability first to avoid training on insufficient data
# - Use lower learning rates (5e-5, 1e-5) for longer training sessions
# - Save checkpoints frequently when training with mixed prompts
# - Consider ensemble approaches by training individual models and combining predictions


# Example 7: Corrected Notebook Cell (Fixed the error)
# ---------------------------------------------------
from src.lora_train_all_prompts import quick_train_all_prompts

# Automatically train with all prompt types that have sufficient data
# This is the corrected version that fixes the TypeError
model = quick_train_all_prompts(
    dataset=dataset,
    strategy="individual",  # or "mixed"
    epochs=10,
    lr=1e-4,
    device="cuda",
    min_images=10,  # Only use prompt types with at least 10 images
    save_final=True,
    checkpoint_interval=5
)

# Alternative: Individual training for specific prompt types
from src.lora_train_all_prompts import train_all_prompt_types_individual

models = train_all_prompt_types_individual(
    dataset=dataset,
    prompt_types=["evi2", "b2b3"],  # Train only EVI2 and B2B3
    epochs=20,
    lr=1e-4,
    device="cuda",
    save_final=True,
    checkpoint_interval=5
)