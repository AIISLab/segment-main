from torchvision import datasets
from torch.utils.data import DataLoader
from collections import Counter
import torch

def _class_counts(ds):
    # ImageFolder stores class index in target
    targets = [y for _, y in ds.samples]
    return Counter(targets)

def get_dataloaders(data_dir, batch_size, img_size, num_workers=4):
    from .transforms import build_transforms
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root=f"{data_dir}/test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    classes = train_ds.classes

    # Optional: class weights for imbalance
    counts = _class_counts(train_ds)
    total = sum(counts.values())
    weights = torch.tensor([total / counts[i] for i in range(len(classes))], dtype=torch.float)
    weights = weights / weights.mean()

    return train_loader, val_loader, test_loader, classes, weights
