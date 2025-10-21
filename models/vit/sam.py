from transformers import SamModel
import torch.nn as nn

def get_sam_model(CFG):
    """
    Build a Segment Anything (SAM) model for downstream fine-tuning or feature extraction.
    Uses the Hugging Face 'facebook/sam-vit-base' backbone by default.
    """

    # Load SAM model (vit-based)
    model = SamModel.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Adjust input channels (SAM expects 3-channel RGB)
    if CFG.in_channels != 3:
        try:
            old_proj = model.vision_encoder.patch_embed.proj
            model.vision_encoder.patch_embed.proj = nn.Conv2d(
                in_channels=CFG.in_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None
            )
        except AttributeError:
            raise NotImplementedError(
                f"[!] Could not locate patch embedding projection in SAM encoder for in_channels={CFG.in_channels}"
            )

    # Optional: Freeze encoder if specified in config
    if getattr(CFG, "freeze_encoder", False):
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

    return model
