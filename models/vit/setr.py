from transformers import AutoModelForSemanticSegmentation
import torch.nn as nn

def get_setr_model(CFG):
    model = AutoModelForSemanticSegmentation.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        num_labels=CFG.num_classes,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )

    # Custom input channels
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
                f"[!] Could not locate patch embedding projection layer in SETR for in_channels={CFG.in_channels}."
            )

    if CFG.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    return model
