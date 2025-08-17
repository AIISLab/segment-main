import torch
from PIL import Image
from pathlib import Path
import sys

import config as CFG
from models.mobilevit import get_mobilevit_s
from utils.transforms import build_transforms

def main(image_path: str):
    ckpt_path = Path(CFG.OUTPUT_DIR) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]

    model = get_mobilevit_s(num_classes=len(classes), in_chans=3, pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(CFG.DEVICE).eval()

    _, eval_tf = build_transforms(CFG.IMG_SIZE)
    img = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(CFG.DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    topk = probs.argsort()[-5:][::-1]
    for i in topk:
        print(f"{classes[i]:<12s}  {probs[i]*100:5.2f}%")

if __name__ == "__main__":
    main(sys.argv[1])
