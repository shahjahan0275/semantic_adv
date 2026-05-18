# ============================================================
# TEST SCRIPT FOR DE-FAKE (CLIP + DCT 4th ORDER STATISTICS)
# ============================================================

from time import process_time_ns
import torch
import clip
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------
# DCT FUNCTION (same as training)
# ---------------------------------------------------------
import torch_dct as dct

def compute_multiband_dct(img):
    """
    img: (3, 224, 224)
    Returns 36-dim multiband DCT statistics.
    """

    c, h, w = img.shape
    dct_img = dct.dct_2d(img)

    LOW = (0, 8)
    MID = (8, 32)
    HIGH = (32, 64)
    bands = [LOW, MID, HIGH]
    feats = []

    for (f1, f2) in bands:
        band = dct_img[:, f1:f2, f1:f2]
        flat = band.reshape(c, -1)

        mean = flat.mean(dim=1)
        var = flat.var(dim=1)
        skew = torch.mean(((flat - mean[:, None]) /
                           torch.sqrt(var[:, None] + 1e-8))**3, dim=1)
        kurt = torch.mean(((flat - mean[:, None]) /
                           torch.sqrt(var[:, None] + 1e-8))**4, dim=1)

        stats = torch.stack([mean, var, skew, kurt], dim=1)
        feats.append(stats)

    feats = torch.cat(feats, dim=1).reshape(-1)
    feats = (feats - feats.mean()) / (feats.std() + 1e-6)
    return feats


# ---------------------------------------------------------
# CLASSIFIER (same as training)
# ---------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


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

        dct_feat = compute_multiband_dct(image)

        return image, self.label, dct_feat


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='DE-FAKE TESTING')
parser.add_argument('--outputpath_clip', type=str, required=True)
parser.add_argument('--outputpath_linear', type=str, required=True)
args = parser.parse_args()


# ---------------------------------------------------------
# LOAD CLIP + MODELS
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

# Classifier: 512-dim CLIP image embedding + 36-dim DCT feature
linear = NeuralNet(input_size=548, hidden_size_list=[512, 256], num_classes=2).to(device)

# Load state_dict files
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
# INFERENCE LOOP
# ---------------------------------------------------------
all_preds = []
all_true = []

with torch.no_grad():
    for x, y, d in tqdm(test_loader):

        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        img_emb = model.encode_image(x)
        dct_emb = d.float()

        emb = torch.cat((img_emb, dct_emb), dim=1)

        logits = linear(emb.float())
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
precision = precision_score(all_true, all_preds)
recall = recall_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds)
accuracy = accuracy_score(all_true, all_preds)

print("\n========= TEST RESULTS =========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("================================\n")
