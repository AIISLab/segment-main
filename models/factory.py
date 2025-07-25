from config import CFG

# Supported architectures
VALID_ARCHS = {
    "segformer": "vit.segformer",
    "setr": "vit.setr",
    "mask2former": "vit.mask2former",
    "unet": "cnn.unet",  # Example CNN fallback
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
    if hasattr(module, f"get_{CFG.architecture}_model"):
        return getattr(module, f"get_{CFG.architecture}_model")(CFG)
    else:
        raise ImportError(f"`get_{CFG.architecture}_model()` not found in {module_path}.py")
