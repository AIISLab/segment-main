import torch
from pathlib import Path
import config as CFG
from utils.dataset import get_dataloaders
from utils.metrics import eval_classification, summarize_report
from models.mobilevit import get_mobilevit_s

def main():
    _, _, test_loader, classes, _ = get_dataloaders(
        CFG.DATA_DIR, CFG.BATCH_SIZE, CFG.IMG_SIZE, CFG.NUM_WORKERS
    )
    ckpt = torch.load(Path(CFG.OUTPUT_DIR) / "best.pt", map_location="cpu")
    model = get_mobilevit_s(num_classes=len(classes), in_chans=3, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(CFG.DEVICE).eval()

    y_true, y_pred = eval_classification(model, test_loader, CFG.DEVICE)
    cm, report = summarize_report(y_true, y_pred, classes)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
