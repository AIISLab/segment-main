import torch
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def eval_classification(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return y_true, y_pred

def summarize_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return cm, report
