import torch
import torch.nn.functional as F
import numpy as np

def one_hot_encode(mask, num_classes, ignore_index=None):
    # Accept numpy or torch; enforce shape [B, H, W] of dtype long
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
    elif mask.ndim != 3:
        raise ValueError(f"Expected mask of shape [B, H, W], but got {tuple(mask.shape)}")
    mask = mask.long()

    # Replace ignored labels BEFORE one_hot so values are in [0, num_classes-1]
    if ignore_index is not None:
        ignore = (mask == ignore_index)
        safe = mask.masked_fill(ignore, 0)  # send ignored to a safe class (0)
    else:
        ignore = None
        safe = mask

    # Validate range early with a clear message
    minv = int(safe.min().item())
    maxv = int(safe.max().item())
    if minv < 0 or maxv >= num_classes:
        uniq = torch.unique(safe).tolist()
        raise ValueError(
            f"Label values must be in [0, {num_classes-1}], got min={minv}, max={maxv}. "
            f"Uniques (sample): {uniq[:20]} "
            f"(Set correct CFG.num_classes or remap labels; ignore_index={ignore_index})"
        )

    one_hot = F.one_hot(safe, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Now zero-out the ignored pixels across all classes
    if ignore_index is not None:
        one_hot *= (~ignore).unsqueeze(1)  # [B,1,H,W] broadcast over classes

    return one_hot


def dice_coef(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if pred.ndim == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)

    pred_one_hot = one_hot_encode(pred, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target, num_classes, ignore_index)

    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Accept logits [B,C,H,W] or labels [B,H,W]
    if pred.ndim == 4 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)

    pred_one_hot = one_hot_encode(pred, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target, num_classes, ignore_index)

    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def evaluate_metrics(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    """
    pred: logits [B, C, H, W] or class predictions [B, H, W]
    target: [B, H, W]
    Returns: dict with accuracy, precision, recall, f1, iou (macro over classes)
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    if pred.ndim == 4 and pred.shape[1] > 1:
        pred_labels = torch.argmax(pred, dim=1)
    else:
        pred_labels = pred.long()

    pred_one_hot = one_hot_encode(pred_labels, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target.long(), num_classes, ignore_index)

    tp = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3)).float()
    fp = (pred_one_hot * (1 - target_one_hot)).sum(dim=(0, 2, 3)).float()
    fn = ((1 - pred_one_hot) * target_one_hot).sum(dim=(0, 2, 3)).float()
    tn = ((1 - pred_one_hot) * (1 - target_one_hot)).sum(dim=(0, 2, 3)).float()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = (2 * precision * recall + smooth) / (precision + recall + smooth)
    accuracy  = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)

    return {
        "accuracy": accuracy.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
    }
