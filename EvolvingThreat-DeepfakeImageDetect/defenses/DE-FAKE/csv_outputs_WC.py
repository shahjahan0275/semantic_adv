import os
import csv
from pathlib import Path

# Base dataset directory
#BASE_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/StyleCLIP_dataset"
#BASE_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet"
#BASE_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet"
BASE_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT"

# Dictionary of all splits
DATASETS = {
    #"train_real": os.path.join(BASE_DIR, "train/real"),
    #"train_fake": os.path.join(BASE_DIR, "train/fake"),
    #"val_real": os.path.join(BASE_DIR, "val/real"),
    #"val_fake": os.path.join(BASE_DIR, "val/fake"),
    #"test_real": os.path.join(BASE_DIR, "test/real"),
    #"test_fake": os.path.join(BASE_DIR, "test/fake"),
    #"CLIPResNet_test_fake": os.path.join(BASE_DIR, "1_fake"),
    #"EfficientNet_test_fake": os.path.join(BASE_DIR, "1_fake"),
    "ViT_test_fake": os.path.join(BASE_DIR, "1_fake")
}

# Output directory for CSV files
OUTPUT_DIR = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs_WC"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed image extensions
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def write_csv(label_name, folder_path):
    """
    Generates a CSV file for the given folder (imagepath only).
    """
    split_name = f"{label_name}.csv"
    csv_path = os.path.join(OUTPUT_DIR, split_name)

    rows = []

    # Get all images in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMG_EXT:
                full_path = os.path.abspath(os.path.join(root, file))
                rows.append([full_path])

    # Write CSV (only one column: imagepath)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imagepath"])
        writer.writerows(rows)

    print(f"Saved: {csv_path}  ({len(rows)} samples)")


# Generate all CSV files
for name, folder in DATASETS.items():
    if not os.path.exists(folder):
        print(f"WARNING: folder not found → {folder}")
        continue
    write_csv(name, folder)
