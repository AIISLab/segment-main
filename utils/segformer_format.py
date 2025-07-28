import argparse
import shutil
from pathlib import Path
from PIL import Image

# Define valid image extensions
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def resize_image(path: Path):
    try:
        img = Image.open(path)
        interp = Image.NEAREST if 'mask' in path.name.lower() else Image.BILINEAR
        img = img.resize((512, 512), interp)
        img.save(path)
    except Exception as e:
        print(f"[!] Failed to resize {path}: {e}")

def main(input_dir):
    input_dir = Path(input_dir).resolve()
    output_dir = input_dir.parent / f"{input_dir.name}_segformer"

    if output_dir.exists():
        print(f"[!] Output directory already exists: {output_dir}")
    else:
        shutil.copytree(input_dir, output_dir)
        print(f"[+] Copied {input_dir} → {output_dir}")

    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            resize_image(path)

    print(f"[✓] All image files in {output_dir} resized to 512x512.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input dataset folder")
    args = parser.parse_args()

    main(args.input_dir)
