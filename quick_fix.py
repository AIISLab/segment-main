#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# Label subfolders
ALLOWED_LABEL_DIRS = ["train_labels", "val_labels", "test_labels"]
# Allowed input formats
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_class_colors(dataset_dir: Path) -> np.ndarray:
    """
    Load allowed class colors from class_dict.csv or class_dictionary.csv.
    Returns an (N, 3) uint8 array of palette colors.
    """
    candidates = [
        dataset_dir / "class_dict.csv",
        dataset_dir / "class_dictionary.csv",
        dataset_dir / Path("class_dict.csv").with_suffix(".CSV"),
        dataset_dir / Path("class_dictionary.csv").with_suffix(".CSV"),
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        sys.exit(f"ERROR: No class_dict.csv or class_dictionary.csv found in {dataset_dir}")

    colors = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]

        if header_lower == ["name", "r", "g", "b"]:
            for row in reader:
                if len(row) >= 4:
                    colors.append([int(row[1]), int(row[2]), int(row[3])])
        elif header_lower == ["r", "g", "b"]:
            for row in reader:
                if len(row) >= 3:
                    colors.append([int(row[0]), int(row[1]), int(row[2])])
        else:
            try:
                idx_r = header_lower.index("r")
                idx_g = header_lower.index("g")
                idx_b = header_lower.index("b")
                for row in reader:
                    if len(row) >= 3:
                        colors.append([int(row[idx_r]), int(row[idx_g]), int(row[idx_b])])
            except ValueError:
                sys.exit(f"ERROR: CSV header must include r,g,b columns. Got: {header}")

    if not colors:
        sys.exit("ERROR: No colors found in CSV.")

    return np.unique(np.array(colors, dtype=np.uint8), axis=0)


def remap_image_to_palette(img: Image.Image, palette: np.ndarray) -> np.ndarray:
    """
    Force every pixel to the nearest allowed RGB in palette.
    Returns RGB image as ndarray.
    """
    img_rgba = img.convert("RGBA")
    arr = np.array(img_rgba)
    rgb = arr[..., :3]

    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.int16)

    pal_int16 = palette.astype(np.int16)
    diff = pal_int16[None, :, :] - flat[:, None, :]
    dist2 = np.sum(diff**2, axis=2)
    nearest_idx = np.argmin(dist2, axis=1)
    remapped_flat = palette[nearest_idx]

    return remapped_flat.reshape(H, W, 3).astype(np.uint8)


def smooth_mask(remapped_rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Convert RGB mask to class indices, run morphological smoothing, convert back to RGB.
    """
    # Map RGB to class index
    index_mask = np.zeros(remapped_rgb.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(palette):
        matches = np.all(remapped_rgb == color, axis=2)
        index_mask[matches] = idx

    # Morphological close then open
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(index_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Map back to RGB
    smoothed_rgb = np.zeros_like(remapped_rgb)
    for idx, color in enumerate(palette):
        smoothed_rgb[opened == idx] = color

    return smoothed_rgb


def process_dir(dataset_dir: Path, palette: np.ndarray) -> dict:
    summary = {"images_total": 0, "images_changed": 0, "pixels_corrected": 0, "files": []}

    for sub in ALLOWED_LABEL_DIRS:
        label_dir = dataset_dir / sub
        if not label_dir.exists():
            continue

        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    with Image.open(p) as im:
                        before = np.array(im.convert("RGBA"))

                        # Remap to nearest palette color
                        remapped_rgb = remap_image_to_palette(im, palette)

                        # Smooth edges
                        smoothed_rgb = smooth_mask(remapped_rgb, palette)

                        # Save as PNG (overwrite if exists)
                        png_path = p.with_suffix(".png")
                        Image.fromarray(smoothed_rgb).save(png_path)

                        after = np.array(Image.fromarray(smoothed_rgb).convert("RGBA"))
                        if not np.array_equal(before, after):
                            diff = np.any(before[..., :3] != after[..., :3], axis=2)
                            n_changed = int(np.sum(diff))
                            summary["images_changed"] += 1
                            summary["pixels_corrected"] += n_changed
                            summary["files"].append(str(png_path.relative_to(dataset_dir)))

                        summary["images_total"] += 1
                except Exception as e:
                    print(f"WARNING: Failed to process {p}: {e}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Convert masks to PNG, remap colors, and smooth edges.")
    ap.add_argument("dataset_dir", type=str, help="Path to dataset folder with labels and CSV.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        sys.exit(f"ERROR: Path does not exist: {dataset_dir}")

    palette = load_class_colors(dataset_dir)
    print(f"Loaded {len(palette)} class color(s): {palette.tolist()}")

    summary = process_dir(dataset_dir, palette)

    print("\nDone.")
    print(f"Images scanned:    {summary['images_total']}")
    print(f"Images modified:   {summary['images_changed']}")
    print(f"Pixels corrected:  {summary['pixels_corrected']}")
    if summary["files"]:
        print("\nModified & saved as PNG:")
        for f in summary["files"]:
            print(f" - {f}")


if __name__ == "__main__":
    main()
