import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50

def get_fcn_model(CFG):
    # Load pretrained FCN model with ResNet-50 backbone
    model = fcn_resnet50(pretrained=True, progress=True)

    # Modify classifier to match number of target classes
    model.classifier[4] = nn.Conv2d(512, CFG.num_classes, kernel_size=1)

    # Modify input channels if not 3
    if CFG.in_channels != 3:
        old_conv = model.backbone.conv1
        model.backbone.conv1 = nn.Conv2d(
            in_channels=CFG.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

    # Optional: Freeze the ResNet encoder
    if hasattr(CFG, "freeze_encoder") and CFG.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
