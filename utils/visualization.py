import numpy as np
from PIL import Image
import os
import torch
import csv

def load_palette_from_csv(csv_path):
    """
    Returns a flat RGB palette list from class_dict.csv for PIL masks.
    Max supported classes: 256 (standard for PIL 'P' mode).
    """
    palette = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row['r'])
            g = int(row['g'])
            b = int(row['b'])
            palette.extend([r, g, b])
    
    # Pad to 256 classes (768 values)
    while len(palette) < 256 * 3:
        palette.extend([0, 0, 0])
    
    return palette

def save_mask(mask_tensor, save_path, palette=None):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask_np, mode="P")
    
    if palette:
        mask_img.putpalette(palette)

    mask_img.save(save_path)


def save_overlay(rgb_tensor, mask_tensor, save_path, palette, alpha=0.5):
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb)

    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    overlay = np.zeros_like(rgb)

    # Map each class to its palette RGB
    for class_idx in np.unique(mask_np):
        r = palette[class_idx * 3]
        g = palette[class_idx * 3 + 1]
        b = palette[class_idx * 3 + 2]
        overlay[mask_np == class_idx] = [r, g, b]

    overlay_img = Image.fromarray(overlay)
    blended = Image.blend(rgb_img, overlay_img, alpha=alpha)
    blended.save(save_path)
