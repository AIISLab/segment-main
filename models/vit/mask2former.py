from transformers import AutoModelForUniversalSegmentation
import torch.nn as nn

def get_mask2former_model(CFG):
    model = AutoModelForUniversalSegmentation.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )

    # Custom input channels (e.g., RGB + thermal)
    if CFG.in_channels != 3:
        try:
            old_proj = model.model.encoder.patch_embed.proj
            model.model.encoder.patch_embed.proj = nn.Conv2d(
                in_channels=CFG.in_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None
            )
        except AttributeError:
            raise NotImplementedError(
                f"[!] Could not find patch embedding projection layer for in_channels={CFG.in_channels} — check your backbone structure."
            )

    # Freeze encoder if requested
    if CFG.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    return model
