import os
import pandas as pd
from config import CFG

def load_class_map(label_csv_path):
    """
    Loads a class map from labels.csv and optionally updates CFG.num_classes.

    Returns:
        class_map (dict): {(r, g, b): class_id}
        class_names (dict): {class_id: class_name}
    """
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Label CSV not found: {label_csv_path}")

    df = pd.read_csv(label_csv_path)

    # Validate required columns
    required = {"class_id", "class_name", "r", "g", "b"}
    if not required.issubset(df.columns):
        raise ValueError(f"labels.csv must contain columns: {required}")

    # Build RGB-to-ID map
    class_map = {
        (row.r, row.g, row.b): int(row.class_id)
        for _, row in df.iterrows()
    }

    # Build ID-to-name map
    class_names = {
        int(row.class_id): row.class_name
        for _, row in df.iterrows()
    }

    # Optionally update CFG.num_classes
    if CFG.num_classes != len(class_names):
        print(f"[INFO] Overriding CFG.num_classes → {len(class_names)}")
        CFG.num_classes = len(class_names)

    return class_map, class_names
