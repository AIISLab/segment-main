# utils/helpers.py

import torch

def get_logits(model_output):
    """
    Normalize model output to raw logits tensor.

    Supports:
        - Hugging Face models with `.logits`
        - Torchvision models returning {'out': logits}
        - Raw tensor output (already logits)

    Raises:
        ValueError: if output format is not recognized.
    """
    if isinstance(model_output, dict) and "out" in model_output:
        return model_output["out"]
    elif hasattr(model_output, "logits"):
        return model_output.logits
    elif isinstance(model_output, torch.Tensor):
        return model_output
    else:
        raise ValueError("Unknown model output format. Expected dict with 'out', tensor, or object with '.logits'.")
