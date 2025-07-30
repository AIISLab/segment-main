import segmentation_models_pytorch as smp
import torch.nn as nn

def get_pspnet_model(CFG):
    # Instantiate PSPNet with a configurable encoder
    model = smp.PSPNet(
        encoder_name=getattr(CFG, "encoder_name", "resnet50"),  # default: resnet50
        encoder_weights="imagenet",                             # pretrained encoder
        in_channels=CFG.in_channels,
        classes=CFG.num_classes,
    )

    # Optional: freeze encoder
    if hasattr(CFG, "freeze_encoder") and CFG.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model
