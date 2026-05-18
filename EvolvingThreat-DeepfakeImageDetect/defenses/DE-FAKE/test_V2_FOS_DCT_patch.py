# ============================================================
# TEST SCRIPT — MATCHED EXACTLY TO TRAINING (DCT + CLIP + 5-LAYER MLP)
# ============================================================

import os
import torch
import clip
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

import torch_dct as dct
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# DCT FEATURE EXTRACTION (MUST MATCH TRAINING EXACTLY)
# ---------------------------------------------------------
def compute_patch_dct_stats(img, num_patches=4):
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

            dct_patch = dct.dct_2d(patch)
            dct_patch = torch.abs(dct_patch)

            patch_feats = []

            for (f1, f2) in bands:
                f2 = min(f2, min(patch_h, patch_w))
                band = dct_patch[:, f1:f2, f1:f2]

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
# MLP MATCHED TO TRAINING — 5 LAYERS
# ---------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size=1088, num_classes=2, dropout_p=0.5):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_p)

        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout_p)

        self.fc5 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        return self.fc5(x)

# ---------------------------------------------------------
# DATASET — EXACTLY MATCHES TRAINING
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

        dct_feat = compute_patch_dct_stats(image)  # 576-D

        return image, self.label, "", dct_feat

# ---------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="DE-FAKE TESTING (DCT)")
parser.add_argument("--outputpath_clip", type=str, required=True)
parser.add_argument("--outputpath_linear", type=str, required=True)
args = parser.parse_args()

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

linear = NeuralNet(
    input_size=1088,
    num_classes=2,
    dropout_p=0.5
).to(device)

# ---- Load CLIP state_dict ONLY ----
state = torch.load(args.outputpath_clip, map_location=device)
if "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=False)

# ---- Load Linear classifier ----
linear.load_state_dict(
    torch.load(args.outputpath_linear, map_location=device)
)

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
    for x, y, _, d in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        img_emb = model.encode_image(x)  # (B, 512)

        # NO DROPOUT IN TESTING!
        emb = torch.cat((img_emb, d), dim=1)  # (B, 1088)

        logits = linear(emb.float())
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y.cpu().numpy())

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds)
recall = recall_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds)

print("\n================= TEST RESULTS (DCT) =================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("======================================================\n")
