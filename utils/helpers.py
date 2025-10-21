import torch
import torch.nn.functional as F

def get_logits(output):
    """
    Universal segmentation output extractor.
    Converts diverse model outputs into logits of shape [B, num_classes, H, W].
    Supports Mask2Former, SAM, SegFormer, FRRN, and plain tensor/dict models.
    """

    # --- Mask2Former (query-based)
    if hasattr(output, "masks_queries_logits") and hasattr(output, "class_queries_logits"):
        masks = output.masks_queries_logits          # [B, Q, H, W]
        class_logits = output.class_queries_logits   # [B, Q, C]
        class_probs = F.softmax(class_logits, dim=-1)
        masks = masks.sigmoid().unsqueeze(-1)
        logits = (masks * class_probs.unsqueeze(2).unsqueeze(3)).sum(1)
        return logits.permute(0, 3, 1, 2).contiguous()

    # --- SAM (Segment Anything)
    if "Sam" in output.__class__.__name__:
        for key in ["low_res_masks", "pred_masks", "masks"]:
            if hasattr(output, key):
                masks = getattr(output, key)
                if isinstance(masks, (list, tuple)):
                    masks = masks[0]

                # ensure 4D tensor: (N, C, H, W)
                if masks.dim() == 2:      # (H, W)
                    masks = masks.unsqueeze(0).unsqueeze(0)
                elif masks.dim() == 3:    # (C, H, W)
                    masks = masks.unsqueeze(0)
                elif masks.dim() == 5:    # sometimes SAM outputs (B, num_masks, 1, H, W)
                    masks = masks.squeeze(2)

                # resize safely to training size
                masks = F.interpolate(masks, size=(512, 512), mode="bilinear", align_corners=False)
                return masks

        print(f"[DEBUG] SAM output attributes: {dir(output)}")
        raise ValueError("Could not locate SAM mask tensor in model output.")

    # --- SegFormer, DeepLab, etc.
    if hasattr(output, "logits"):
        return output.logits

    # --- FRRN / torchvision-like dicts
    if isinstance(output, dict):
        if "out" in output:
            return output["out"]
        elif "logits" in output:
            return output["logits"]

    # --- Direct tensor
    if isinstance(output, torch.Tensor):
        return output

    raise ValueError(f"Unknown model output type: {type(output)}")
