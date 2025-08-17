import random, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

import config as CFG
from utils.dataset import get_dataloaders
from models.mobilevit import get_mobilevit_s, freeze_stages

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    set_seed(CFG.SEED)
    train_loader, val_loader, _, classes, class_weights = get_dataloaders(
        CFG.DATA_DIR, CFG.BATCH_SIZE, CFG.IMG_SIZE, CFG.NUM_WORKERS
    )
    num_classes = len(classes)

    model = get_mobilevit_s(num_classes=num_classes, in_chans=3, pretrained=True)
    freeze_stages(model, CFG.FREEZE_STAGES)
    model.to(CFG.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTH,
                                    weight=class_weights.to(CFG.DEVICE))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    scaler = GradScaler()

    best_acc, patience = 0.0, 0
    ckpt_path = Path(CFG.OUTPUT_DIR) / "best.pt"

    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS}")

        for imgs, targets in pbar:
            imgs = imgs.to(CFG.DEVICE, non_blocking=True)
            targets = targets.to(CFG.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        # Validation
        val_acc = evaluate_top1(model, val_loader, CFG.DEVICE)
        scheduler.step()

        # Early stopping + checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": {
                    "img_size": CFG.IMG_SIZE,
                    "model_name": CFG.MODEL_NAME
                }
            }, ckpt_path)
        else:
            patience += 1
            if patience >= CFG.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: val_acc={val_acc:.4f} | best={best_acc:.4f}")

    print(f"Best checkpoint saved to: {ckpt_path}")

@torch.no_grad()
def evaluate_top1(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return correct / max(total, 1)

if __name__ == "__main__":
    main()
