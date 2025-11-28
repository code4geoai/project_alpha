import matplotlib.pyplot as plt
import numpy as np

def show_overlays(rgb, mask, pred):
    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.title("RGB")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='gray')
    plt.axis("off")

    plt.show()
