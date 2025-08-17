from pathlib import Path
import torch

DATA_DIR = Path("cls_data")
OUTPUT_DIR = Path("runs/mobilevit_s")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 288           # MobileViT works well 256–320; 288 is a sweet spot
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTH = 0.05
NUM_WORKERS = 4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "mobilevit_s"   # timm id
FREEZE_STAGES = 0            # set >0 if you want to freeze early layers
EARLY_STOP_PATIENCE = 7
