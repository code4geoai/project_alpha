import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import torch.nn.functional as F

#from segment_anything import sam_model_registry, SamPredictor
#from src.lora_sam import load_sam_with_lora
#from src.step2_scaling import load_nc_image
#from src.config import SAM_MODEL_TYPE, SAM_CHECKPOINT


# -------------------------
# Metrics
# -------------------------
def compute_iou(pred, gt):
    inter = ((pred > 0.5) & (gt > 0.5)).sum()
    union = ((pred > 0.5) | (gt > 0.5)).sum()
    return (inter / (union + 1e-6)).item()


def compute_dice(pred, gt):
    inter = (pred * gt).sum()
    return (2 * inter / (pred.sum() + gt.sum() + 1e-6)).item()


def compute_boundary_f1(pred, gt):
    sobel = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                         device=pred.device).float().view(1,1,3,3)

    def grad(x):
        gx = F.conv2d(x, sobel, padding=1)
        gy = F.conv2d(x, sobel.transpose(2,3), padding=1)
        return torch.sqrt(gx**2 + gy**2)

    pb = grad(pred.unsqueeze(0).unsqueeze(0))
    gb = grad(gt.unsqueeze(0).unsqueeze(0))

    return F.l1_loss(pb, gb).item()
