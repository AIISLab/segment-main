import torch
import torch.nn.functional as F

def one_hot_encode(mask, num_classes, ignore_index=None):
    B, H, W = mask.shape
    one_hot = F.one_hot(mask.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if ignore_index is not None:
        ignore_mask = (mask == ignore_index).unsqueeze(1)
        one_hot *= ~ignore_mask
    return one_hot

def dice_coef(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred_one_hot = one_hot_encode(pred, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target, num_classes, ignore_index)

    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred_one_hot = one_hot_encode(pred, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target, num_classes, ignore_index)

    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    union = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def evaluate_metrics(pred, target, num_classes, ignore_index=None, smooth=1e-6):
    """
    pred: logits [B, C, H, W]
    target: [B, H, W]
    Returns: dict with accuracy, precision, recall, f1, iou
    """
    pred_labels = torch.argmax(pred, dim=1)  # [B, H, W]

    pred_one_hot = one_hot_encode(pred_labels, num_classes, ignore_index)
    target_one_hot = one_hot_encode(target, num_classes, ignore_index)

    tp = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3)).float()
    fp = (pred_one_hot * (1 - target_one_hot)).sum(dim=(0, 2, 3)).float()
    fn = ((1 - pred_one_hot) * target_one_hot).sum(dim=(0, 2, 3)).float()
    tn = ((1 - pred_one_hot) * (1 - target_one_hot)).sum(dim=(0, 2, 3)).float()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = (2 * precision * recall + smooth) / (precision + recall + smooth)
    accuracy  = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    return {
        "accuracy": accuracy.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
    }
