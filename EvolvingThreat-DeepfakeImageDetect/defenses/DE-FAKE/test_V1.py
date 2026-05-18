from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import argparse
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------
# SEEDING
# ---------------------------------------------------------
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


# ---------------------------------------------------------
# CLASSIFIER MODEL
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
# DATASETS
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
        caption = str(self.data.iloc[idx]["caption"])
        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image)
        return image, self.label, caption


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='DE-FAKE Inference')
parser.add_argument('--outputpath_clip', type=str, required=True,
                    help='Path to finetuned CLIP model .pt')
parser.add_argument('--outputpath_linear', type=str, required=True,
                    help='Path to finetuned linear classifier .pt')
args = parser.parse_args()


# ---------------------------------------------------------
# LOAD CLIP + MODELS
# ---------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize fresh classifier
linear = NeuralNet(1024, [512, 256], 2).to(device)

# Load saved weights (state_dict)
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
fake_test = CustomDataset("/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/test_fake.csv", label=1)

#fake_test = CustomDataset("/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/CLIPResNet_test_fake.csv", label=1)
#fake_test = CustomDataset("/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/EfficientNet_test_fake.csv", label=1)
#fake_test = CustomDataset("/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/ViT_test_fake.csv", label=1)


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
    for x, y, captions in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)

        # Encode text
        text_tokens = clip.tokenize(list(captions), truncate=True).to(device)
        text_emb = model.encode_text(text_tokens)

        # Encode images
        img_emb = model.encode_image(x)

        # Concatenate image+text embedding
        emb = torch.cat((img_emb, text_emb), dim=1)

        # Classifier prediction
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
print("================================")
