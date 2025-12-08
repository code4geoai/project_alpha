import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from segment_anything import SamPredictor, sam_model_registry
from src.config import SAM_CHECKPOINT, SAM_MODEL_TYPE

def load_sam_model(device="cuda"):
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def sam_predict_from_points(predictor, image, points):
    predictor.set_image((image * 255).astype(np.uint8))
    masks = []
    for pt in points:
        m, _, _ = predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        masks.append(m[0])
    return masks

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask > 0).sum()
    union = np.logical_or(pred_mask, gt_mask > 0).sum()
    return intersection / union if union > 0 else 0

def binarize_mask(mask):
    return (mask > 0.5).astype(np.uint8)
