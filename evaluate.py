import argparse
import os
import torch
import torch.nn.functional as F
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import iou_score
from utils.helpers import get_logits
from utils.cli import parse_args
from tqdm import tqdm
import numpy as np
from datetime import datetime
from utils.visualization import save_mask, save_overlay, load_palette_from_csv

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------ CLI ARGUMENTS ------------------
args = parse_args()
CFG.architecture = args.architecture
CFG.model_name = args.model_name
CFG.dataset_root = args.data_root
CFG.label_csv = args.label_csv
CFG.weights = args.weights

# ------------------ SETUP ------------------

torch.manual_seed(CFG.seed)
device = CFG.device

model = get_model().to(device)
model.load_state_dict(torch.load(CFG.weights, map_location=device))
print("[INFO] Loaded model weights from:", CFG.weights)
model.eval()

_, val_loader = get_loaders(CFG.dataset_root, CFG.label_csv)
csv_path = os.path.join(CFG.dataset_root, CFG.label_csv)
palette = load_palette_from_csv(csv_path)

# Create output folders
# Generate date string
today = datetime.now().strftime("%Y-%m-%d")

# Build output paths
mask_dir = os.path.join(CFG.output_dir, CFG.architecture, today, "masks")
overlay_dir = os.path.join(CFG.output_dir, CFG.architecture, today, "overlays")
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

# ------------------ EVALUATION ------------------

flat_preds = []
flat_targets = []
all_preds_tensor = []
all_targets_tensor = []

sample_count = 0

with torch.no_grad():
    for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
        images, masks = images.to(device), masks.to(device)
        outputs = get_logits(model(images))
        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = outputs.argmax(dim=1)  # [B, H, W]

        flat_preds.extend(preds.cpu().numpy().reshape(-1))
        flat_targets.extend(masks.cpu().numpy().reshape(-1))

        all_preds_tensor.append(preds.cpu())
        all_targets_tensor.append(masks.cpu())

        # Save example masks and overlays
        if CFG.show_sample_predictions and sample_count < CFG.num_eval_samples:
            for b in range(images.size(0)):
                image_idx = sample_count
                save_mask(preds[b], os.path.join(mask_dir, f"sample_{image_idx}_mask.png"), palette)
                save_overlay(images[b], preds[b], os.path.join(overlay_dir, f"sample_{image_idx}_overlay.png"), palette)
                sample_count += 1
                if sample_count >= CFG.num_eval_samples:
                    break

# ------------------ FLATTENED METRICS ------------------

flat_preds = np.array(flat_preds)
flat_targets = np.array(flat_targets)
mask = flat_targets != CFG.ignore_index
flat_preds = flat_preds[mask]
flat_targets = flat_targets[mask]

acc = accuracy_score(flat_targets, flat_preds)
prec = precision_score(flat_targets, flat_preds, average="macro", zero_division=0)
rec = recall_score(flat_targets, flat_preds, average="macro", zero_division=0)
f1 = f1_score(flat_targets, flat_preds, average="macro", zero_division=0)

# ------------------ TENSOR METRICS ------------------

all_preds_tensor = torch.cat(all_preds_tensor, dim=0)
all_targets_tensor = torch.cat(all_targets_tensor, dim=0)

iou = iou_score(all_preds_tensor, all_targets_tensor, CFG.num_classes, ignore_index=CFG.ignore_index)

# ------------------ RESULTS ------------------

print("\n[Evaluation Results]")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
