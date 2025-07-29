import argparse
import os
import torch
import torch.nn.functional as F
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import iou_score
from tqdm import tqdm
import numpy as np

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------ CLI ARGUMENTS ------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default=CFG.architecture)
    parser.add_argument("--model_name", type=str, default=CFG.model_name)
    parser.add_argument("--data_root", type=str, default=CFG.dataset_root)
    parser.add_argument("--label_csv", type=str, default=CFG.label_csv)
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    return parser.parse_args()

args = parse_args()
CFG.architecture = args.architecture
CFG.model_name = args.model_name
CFG.dataset_root = args.data_root
CFG.label_csv = args.label_csv

# ------------------ SETUP ------------------

torch.manual_seed(CFG.seed)
device = CFG.device

model = get_model(CFG).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

_, val_loader = get_loaders(CFG.dataset_root, CFG.label_csv)

# ------------------ EVALUATION ------------------

all_preds = []
all_targets = []

with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images).logits
        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = outputs.argmax(dim=1)  # shape: [B, H, W]

        all_preds.extend(preds.cpu().numpy().reshape(-1))
        all_targets.extend(masks.cpu().numpy().reshape(-1))

# ------------------ METRICS ------------------

# Filter out ignore index
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
mask = all_targets != CFG.ignore_index
all_preds = all_preds[mask]
all_targets = all_targets[mask]

acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
iou = iou_score(all_preds, all_targets, CFG.num_classes)

print("\n[Evaluation Results]")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
