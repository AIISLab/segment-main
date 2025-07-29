MODEL_ZOO = {
    "segformer": {
        "default_model": "nvidia/segformer-b3-finetuned-ade-512-512",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": False,
        "image_size": (512,512),
    },
    "setr": {
        "default_model": "damo/SETR_MLA",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": True,
        "image_size": (512,512),

    },
    "mask2former": {
        "default_model": "shi-labs/mask2former-swin-large-ade",
        "num_classes": 150,
        "in_channels": 3,
        "trust_remote_code": True,
        "image_size": (512,512),
    }
}
