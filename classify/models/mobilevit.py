import timm
import torch.nn as nn

def get_mobilevit_s(num_classes: int, in_chans: int = 3, pretrained: bool = True):
    # timm id: 'mobilevit_s' (Apple MobileViT Small)
    model = timm.create_model(
        "mobilevit_s",
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )
    return model

def freeze_stages(model: nn.Module, n_stages: int = 0):
    """
    Freeze early stages. For MobileViT, stages map roughly to stem + stages.
    You can freeze by name prefix.
    """
    if n_stages <= 0:
        return
    prefixes = [
        "conv_stem",   # stage 1
        "blocks.0",    # stage 2
        "blocks.1",    # stage 3
        "blocks.2",    # stage 4
        "blocks.3",    # stage 5
    ]
    for i in range(min(n_stages, len(prefixes))):
        for n, p in model.named_parameters():
            if n.startswith(prefixes[i]):
                p.requires_grad = False
