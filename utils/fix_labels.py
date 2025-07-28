import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def load_classes(class_csv_path):
    df = pd.read_csv(class_csv_path)
    return [tuple(map(int, (row.r, row.g, row.b))) for _, row in df.iterrows()]

def closest_color(pixel, class_colors):
    distances = [np.linalg.norm(np.array(pixel) - np.array(c)) for c in class_colors]
    return class_colors[np.argmin(distances)]

def process_mask(mask_path, class_colors):
    mask = Image.open(mask_path).convert("RGB")
    np_mask = np.array(mask)
    h, w, _ = np_mask.shape

    reshaped = np_mask.reshape(-1, 3)
    fixed = np.zeros_like(reshaped)

    changed = False
    for i, px in enumerate(reshaped):
        px_tuple = tuple(px)
        if px_tuple not in class_colors:
            closest = closest_color(px_tuple, class_colors)
            fixed[i] = closest
            changed = True
        else:
            fixed[i] = px

    fixed_mask = fixed.reshape(h, w, 3)
    Image.fromarray(fixed_mask.astype(np.uint8)).save(mask_path)

    return changed

def fix_labels(label_dirs, class_csv):
    class_colors = load_classes(class_csv)

    for label_dir in label_dirs:
        print(f"🛠️ Processing: {label_dir}")
        mask_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for fname in tqdm(mask_files, desc=f"Fixing {label_dir}"):
            path = os.path.join(label_dir, fname)
            changed = process_mask(path, class_colors)
            if changed:
                print(f"🔧 {fname} had mismatched pixels — fixed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dirs", nargs="+", required=True, help="Paths to label folders")
    parser.add_argument("--class_csv", required=True, help="Path to class_dict.csv")
    args = parser.parse_args()

    fix_labels(args.label_dirs, args.class_csv)
