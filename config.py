from types import SimpleNamespace
import torch

CFG = SimpleNamespace(
    # General
    project_name = "segmentation-pipeline",
    seed = 42,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Model (default)
    architecture = "segformer",
    model_name = "nvidia/segformer-b2-finetuned-ade-512-512",
    num_classes = 2,
    ignore_index = 255,
    pretrained = True,
    freeze_encoder = False,

    # Input
    image_size = (512, 512),
    in_channels = 3,

    # Training
    epochs = 20,
    batch_size = 8,
    learning_rate = 5e-5,
    weight_decay = 1e-4,
    val_every = 1,

    # Loss
    use_dice_loss = False,
    dice_weight = 0.5,

    # Data paths
    dataset_root = "data",  # default, can be overridden via CLI
    label_csv = "class_dict.csv",  # class-color mapping CSV inside dataset root

    # Logging / Outputs
    output_dir = "./results/",
    save_best_only = True,
    log_dir = "./logs/",

    # Evaluation
    show_sample_predictions = True,
    num_eval_samples = 5
)
