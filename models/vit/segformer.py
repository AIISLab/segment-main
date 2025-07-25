import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers import TrainingArguments, Trainer

# --------------- CONFIG ---------------
NUM_CLASSES = 2  # sunlit vs shaded
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------- DATASET ---------------
class SunlitSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, image_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.image_filenames = os.listdir(image_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size)
        mask = mask.resize(self.image_size, resample=Image.NEAREST)

        # Normalize image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}

        # Convert mask to tensor
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = (mask > 127).long()  # convert to binary (0 or 1)

        inputs["labels"] = mask
        return inputs

# --------------- MAIN PIPELINE ---------------
def main():
    # Paths
    train_images = "./data/train/images"
    train_masks = "./data/train/masks"
    val_images = "./data/val/images"
    val_masks = "./data/val/masks"

    # Feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = SunlitSegmentationDataset(train_images, train_masks, feature_extractor)
    val_dataset = SunlitSegmentationDataset(val_images, val_masks, feature_extractor)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    # Metrics stub
    def compute_metrics(eval_pred):
        return {}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained("./segformer-sunlit")

if __name__ == "__main__":
    main()
