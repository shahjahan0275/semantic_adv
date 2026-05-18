"""
Builds the full 2048-D reference set (no dimensionality reduction)
from known fake training images using the trained PDA detector.

Usage example:
python build_reference_set_full.py \
    --detector-path /path/to/detector.pth \
    --ref-fake /path/to/fake_train_folder \
    --save-path reference_set_full.npz \
    --device cuda:0
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------------------------------------------------
# Detector (same as training)
# ----------------------------------------------------------------------
class Detector(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=False):
        super().__init__()
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=None if not pretrained else models.ResNet50_Weights.DEFAULT
            )
            in_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(in_feat, 1)
            self.feat_dim = in_feat
        else:
            raise ValueError("Unsupported backbone")

    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x)

# ----------------------------------------------------------------------
# Dataset loader
# ----------------------------------------------------------------------
class ImageFolderList(Dataset):
    def __init__(self, files, size=224):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path

# ----------------------------------------------------------------------
# Activation pruning (same as training)
# ----------------------------------------------------------------------
def activation_prune(batch_feats: np.ndarray, percentile: float = 90.0):
    pruned = []
    for x in batch_feats:
        c = np.percentile(x, percentile)
        pruned.append(np.minimum(x, c))
    return np.stack(pruned, axis=0)

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def find_files(folder, exts=(".png", ".jpg", ".jpeg")):
    folder = Path(folder)
    files = [str(x) for x in folder.rglob("*") if x.suffix.lower() in exts]
    files.sort()
    return files

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(detector_path, ref_fake_dir, save_path, device="cuda:0", batch_size=32, size=224):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load detector
    detector = Detector(backbone_name="resnet50", pretrained=False).to(device)
    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.eval()
    print(f"✅ Loaded detector from: {detector_path}")

    # Gather files
    ref_files = find_files(ref_fake_dir)
    print(f"Found {len(ref_files)} fake reference images in: {ref_fake_dir}")

    # Dataset + DataLoader
    ds = ImageFolderList(ref_files, size=size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats, all_paths = [], []

    with torch.no_grad():
        for xb, paths in tqdm(dl, desc="Extracting 2048-D features"):
            xb = xb.to(device)
            feats = detector.extract_features(xb)
            feats = feats.cpu().numpy()
            all_feats.append(feats)
            all_paths.extend(paths)

    feats = np.concatenate(all_feats, axis=0)
    print(f"[INFO] Extracted features shape: {feats.shape}")

    pruned = activation_prune(feats, percentile=90.0)
    print(f"[INFO] After pruning: {pruned.shape}")

    os.makedirs(Path(save_path).parent, exist_ok=True)
    np.savez_compressed(save_path, pruned=pruned, paths=all_paths)
    print(f"✅ Saved 2048-D reference set to: {save_path}")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-path", type=str, required=True,
                        help="Path to trained detector .pth")
    parser.add_argument("--ref-fake", type=str, required=True,
                        help="Folder containing known fake images (train set)")
    parser.add_argument("--save-path", type=str, default="reference_set_full.npz",
                        help="Path to save the output reference set npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)

    args = parser.parse_args()
    main(args.detector_path, args.ref_fake, args.save_path,
         args.device, args.batch_size, args.size)
