# ============================================================
# TEST SCRIPT FOR DE-FAKE (CLIP + PATCH FFT 4th-ORDER STATS)
# ============================================================

import torch
import clip
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------
# PATCH-BASED FFT 4TH ORDER STATS (same as training)
# ---------------------------------------------------------
def compute_patch_fft_stats(img, num_patches=4):
    C, H, W = img.shape
    patch_h = H // num_patches
    patch_w = W // num_patches

    LOW = (0, 16)
    MID = (16, 48)
    HIGH = (48, 80)
    bands = [LOW, MID, HIGH]

    features = []

    for py in range(num_patches):
        for px in range(num_patches):

            patch = img[:, py*patch_h:(py+1)*patch_h,
                        px*patch_w:(px+1)*patch_w]

            fft = torch.fft.fft2(patch)
            fft = torch.abs(fft)

            patch_feats = []

            for (f1, f2) in bands:
                f2 = min(f2, min(patch_h, patch_w))

                band = fft[:, f1:f2, f1:f2]
                flat = band.reshape(C, -1)

                mean = flat.mean(dim=1)
                var = flat.var(dim=1)
                std = torch.sqrt(var + 1e-8)
                z = (flat - mean[:, None]) / std[:, None]

                skew = torch.mean(z**3, dim=1)
                kurt = torch.mean(z**4, dim=1)

                stats = torch.stack([mean, var, skew, kurt], dim=1)
                patch_feats.append(stats)

            patch_feats = torch.cat(patch_feats, dim=1).reshape(-1)
            features.append(patch_feats)

    features = torch.cat(features, dim=0)
    features = (features - features.mean()) / (features.std() + 1e-6)
    return features  # 576-D


# ---------------------------------------------------------
# CLASSIFIER (same as training)
# ---------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super().__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------
# DATASET (same as training)
# ---------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_file, label):
        self.data = pd.read_csv(csv_file)
        self.label = label
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert("RGB")
        image = self.transform(image).float()

        fft_feat = compute_patch_fft_stats(image)  # 576-dim
        return image, self.label, fft_feat


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="DE-FAKE TESTING")
parser.add_argument("--outputpath_clip", type=str, required=True)
parser.add_argument("--outputpath_linear", type=str, required=True)
args = parser.parse_args()


# ---------------------------------------------------------
# LOAD CLIP + CLASSIFIER
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
linear = NeuralNet(input_size=1088, hidden_size_list=[512, 256], num_classes=2).to(device)

# Load weights
clip_state = torch.load(args.outputpath_clip, map_location=device)
linear_state = torch.load(args.outputpath_linear, map_location=device)

model.load_state_dict(clip_state)
linear.load_state_dict(linear_state)

model.eval()
linear.eval()


# ---------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------
real_test = CustomDataset(
    "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/test_real.csv",
    label=0
)
fake_test = CustomDataset(
    "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/test_fake.csv",
    label=1
)

test_dataset = ConcatDataset([real_test, fake_test])

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------
all_preds = []
all_true = []

with torch.no_grad():
    for x, y, d in tqdm(test_loader):

        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        img_emb = model.encode_image(x)         # (B, 512)
        emb = torch.cat((img_emb, d), dim=1)    # (B, 1088)

        logits = linear(emb.float())
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds)
recall = recall_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds)

print("\n================= TEST RESULTS =================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("================================================\n")
