"""
ViT model implementations for semantic segmentation.

Each file in this package should implement:
    def get_<architecture>_model(CFG):
        -> returns a torch.nn.Module ready for training/evaluation.

Expected filenames and entry points:
    - segformer.py     -> get_segformer_model()
    - setr.py          -> get_setr_model()
    - mask2former.py   -> get_mask2former_model()
"""

# Optional: expose available architectures here (not required if using dynamic imports)
__all__ = [
    "segformer",
    "setr",
    "mask2former"
]
