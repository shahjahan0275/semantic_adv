import os
import torch
import torch_dct as dct
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, average_precision_score, confusion_matrix
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------
# SAME LOGISTIC REGRESSION MODEL AS TRAINING
# ------------------------------------------------------
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# *** IDENTICAL spectral_features() TO TRAINING ***
# Multi-band DCT with 4th-order stats
# ------------------------------------------------------
def spectral_features(img_batch, patch=32):
    """
    Multi-band DCT, 4th-order spectral statistics.
    Output = Npatches × 12 (mean, var, skew, kurt) for 3 bands.
    """

    # ----------- Convert to grayscale -----------
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(1)
    elif img_batch.shape[1] == 3:
        r, g, b = img_batch[:, 0], img_batch[:, 1], img_batch[:, 2]
        img_batch = (0.299*r + 0.587*g + 0.114*b).unsqueeze(1)

    img_batch = img_batch.double()
    B, C, H, W = img_batch.shape
    device = img_batch.device

    # ------------------ 2D DCT ------------------
    with torch.no_grad():
        F = dct.dct_2d(img_batch, norm="ortho").squeeze(1)
        F = F.abs()

    # ------------------ Patchify ------------------
    ph = pw = patch
    patches = F.unfold(1, ph, ph).unfold(2, pw, pw)
    B, Hp, Wp, ph, pw = patches.shape
    N = Hp * Wp
    patches = patches.reshape(B, N, ph, pw)

    # ------------------ Radial masks ------------------
    yy, xx = torch.meshgrid(
        torch.arange(ph, device=device),
        torch.arange(pw, device=device),
        indexing="ij"
    )
    cy = (ph - 1) / 2
    cx = (pw - 1) / 2

    dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    dist = dist / dist.max()

    low_mask  = (dist <= 0.25).double()[None, None, :, :]
    mid_mask  = ((dist > 0.25) & (dist <= 0.65)).double()[None, None, :, :]
    high_mask = (dist > 0.65).double()[None, None, :, :]

    eps = 1e-6

    def compute_stats(mask):
        """
        Returns [B, N, 4] (mean, variance, skew, kurtosis)
        """
        X = patches * mask
        cnt = mask.sum()

        s1 = X.sum(dim=[2, 3]) + eps
        mean = s1 / cnt

        xc = X - mean[:, :, None, None]
        var = (xc * xc).sum(dim=[2, 3]) / cnt

        skew = ((xc**3).sum(dim=[2, 3]) / cnt) / (var.sqrt()**3 + eps)
        kurt = ((xc**4).sum(dim=[2, 3]) / cnt) / (var**2 + eps)

        return torch.stack([mean, var, skew, kurt], dim=-1)

    low_s  = compute_stats(low_mask)
    mid_s  = compute_stats(mid_mask)
    high_s = compute_stats(high_mask)

    feat = torch.cat([low_s, mid_s, high_s], dim=-1)   # [B, N, 12]
    feat = feat.reshape(B, -1).float()

    return feat


# ------------------------------------------------------
# Image loader (same preprocessing as training)
# ------------------------------------------------------
def load_img(path, input_size=1024):

    try:
        img = Image.open(path)
        img = img.convert("L")
        img = transforms.CenterCrop((input_size, input_size))(img)
        img = transforms.ToTensor()(img)
        img = (img * 2.0) - 1.0
        return img

    except Exception:
        return None


# ------------------------------------------------------
# TEST PIPELINE
# ------------------------------------------------------
def run_test(args):

    # ==== LOAD NORMALIZATION =====
    means = torch.load(os.path.join(args.meanstd_dir, "means.pt"), map_location=DEVICE)
    stds  = torch.load(os.path.join(args.meanstd_dir, "stds.pt"), map_location=DEVICE)

    input_size = means.shape[1]
    print("Loaded feature dimension =", input_size)

    # ==== LOAD MODEL ====
    model = LogisticRegression(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # ==== GATHER FILES ====
    real_paths = sorted([str(Path(args.real_root) / f) for f in os.listdir(args.real_root)])
    fake_paths = sorted([str(Path(args.fake_root) / f) for f in os.listdir(args.fake_root)])

    all_paths = real_paths + fake_paths
    all_labels = [0]*len(real_paths) + [1]*len(fake_paths)

    y_true, y_pred, y_scores = [], [], []
    csv_rows = []

    print("\nProcessing Test Images...")
    for path, label in tqdm(list(zip(all_paths, all_labels))):

        img = load_img(path, args.input_size)
        if img is None:
            continue

        img = img.unsqueeze(0).to(DEVICE)

        feat = spectral_features(img)
        feat = (feat - means) / stds

        with torch.no_grad():
            out = model(feat)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            pred = torch.argmax(out, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(prob)

        csv_rows.append([path, label, pred])

    # ===== METRICS =====
    ACC = accuracy_score(y_true, y_pred)
    REC = recall_score(y_true, y_pred)
    PRE = precision_score(y_true, y_pred)
    F1  = f1_score(y_true, y_pred)
    AP  = average_precision_score(y_true, y_scores)
    CM  = confusion_matrix(y_true, y_pred)

    print("\n========== TEST METRICS ==========")
    print(f"Accuracy  : {ACC:.4f}")
    print(f"Recall    : {REC:.4f}")
    print(f"Precision : {PRE:.4f}")
    print(f"F1 score  : {F1:.4f}")
    print(f"AP        : {AP:.4f}")
    print("\nConfusion Matrix:\n", CM)
    print("===================================\n")

    # ===== SAVE CSV =====
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "prediction"])
        w.writerows(csv_rows)

    print("Saved test results →", args.out_csv)


# ------------------------------------------------------
# ARGS
# ------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fake_root", type=str, required=True,
                        help="Path to 1_fake folder")

    parser.add_argument("--real_root", type=str, required=True,
                        help="Path to 0_real folder")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth")

    parser.add_argument("--meanstd_dir", type=str, required=True,
                        help="Folder containing means.pt and stds.pt")

    parser.add_argument("--input_size", type=int, default=1024)

    parser.add_argument("--out_csv", type=str, default="test_results.csv")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
