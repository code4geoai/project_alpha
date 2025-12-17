# src/config.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PROMPTS_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "prompts"))
REFINED_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "prompts_refined"))
TEMPORAL_PROMPTS_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "temporal_prompts"))
RANKED_PROMPTS_DIR   = os.path.abspath(os.path.join(RESULTS_DIR, "temporal_prompts_ranked"))
PRED_MASK_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "predicted_masks"))
PRED_VIS_DIR = os.path.abspath(os.path.join(RESULTS_DIR, "pred_vis"))
SUMMARY_CSV = os.path.abspath(os.path.join(RESULTS_DIR, "sam_summary.csv"))

os.makedirs(TEMPORAL_PROMPTS_DIR, exist_ok=True)
os.makedirs(RANKED_PROMPTS_DIR, exist_ok=True)

# SAM
SAM_CHECKPOINT = os.path.join(BASE_DIR, "models", "sam_vit_h.pth")
SAM_MODEL_TYPE = "vit_h"

# Processing settings
SLIC_SEGMENTS = 70     # can tune later
NDVI_THRESHOLD = 0.2   # weak vegetation mask


