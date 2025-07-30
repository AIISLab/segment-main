from config import CFG

# Supported architectures
VALID_ARCHS = {
    # ViT-based
    "segformer": "vit.segformer",
    "setr": "vit.setr",
    "mask2former": "vit.mask2former",
    
    # CNN-based
    "deeplabv3": "cnn.deeplabv3",
    "fcn": "cnn.fcn",
    "pspnet": "cnn.pspnet",
}

def get_model():
    if CFG.architecture not in VALID_ARCHS:
        raise ValueError(
            f"Invalid architecture '{CFG.architecture}'. "
            f"Must be one of: {list(VALID_ARCHS.keys())}"
        )

    # Dynamically import from correct submodule
    module_path = VALID_ARCHS[CFG.architecture]
    module = __import__(f"models.{module_path}", fromlist=["get_model_func"])

    # Convention: every model file defines a get_<model>_model(config) function
    func_name = f"get_{CFG.architecture}_model"
    if hasattr(module, func_name):
        return getattr(module, func_name)(CFG)
    else:
        raise ImportError(f"`{func_name}()` not found in {module_path}.py")
