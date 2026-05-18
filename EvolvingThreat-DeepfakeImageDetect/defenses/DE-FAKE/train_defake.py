import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import clip
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# =====================
# CSV Dataset (image only)
# =====================
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None, label_value=0):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_value = label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["imagepath"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.label_value


# =====================
# Classifier on top of CLIP image embeddings
# =====================
class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =====================
# Evaluation using image embeddings ONLY
# =====================
def evaluate(model, linear, loader, device):
    model.eval()
    linear.eval()

    preds, gts = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            img_emb = model.encode_image(images)
            outputs = linear(img_emb.float())
            _, pred = torch.max(outputs, 1)

            preds.extend(pred.cpu().numpy())
            gts.extend(targets.cpu().numpy())

    acc = accuracy_score(gts, preds)
    rec = recall_score(gts, preds, average="weighted")
    pre = precision_score(gts, preds, average="weighted")
    f1  = f1_score(gts, preds, average="weighted")

    return acc, rec, pre, f1


# =====================
# Main
# =====================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Load datasets
    train_real = CSVDataset(args.train_real, transform, label_value=0)
    train_fake = CSVDataset(args.train_fake, transform, label_value=1)
    val_real = CSVDataset(args.val_real, transform, label_value=0)
    val_fake = CSVDataset(args.val_fake, transform, label_value=1)

    test_real = CSVDataset(args.test_real, transform, label_value=0) if args.test_real else None
    test_fake = CSVDataset(args.test_fake, transform, label_value=1) if args.test_fake else None

    train_dataset = torch.utils.data.ConcatDataset([train_real, train_fake])
    val_dataset = torch.utils.data.ConcatDataset([val_real, val_fake])
    test_dataset = torch.utils.data.ConcatDataset([test_real, test_fake]) if test_real and test_fake else None

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4) if test_dataset else None

    # Load CLIP image encoder
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()  # ALWAYS freeze CLIP

    # Freeze CLIP weights
    for p in model.parameters():
        p.requires_grad = False

    # Linear classifier (takes 512-dim embeddings)
    linear = Classifier(input_dim=512).to(device)

    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # =====================
    # Training
    # =====================
    for epoch in range(args.epochs):
        linear.train()

        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                img_emb = model.encode_image(images)

            outputs = linear(img_emb.float())
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        acc, rec, pre, f1 = evaluate(model, linear, val_loader, device)
        print(f"[Epoch {epoch}] Val Acc: {acc:.4f}, Recall: {rec:.4f}, Precision: {pre:.4f}, F1: {f1:.4f}")

    # save models
    torch.save(linear.state_dict(), args.output_linear)
    print(f"Saved classifier → {args.output_linear}")

    # =====================
    # Test Evaluation
    # =====================
    if test_loader:
        print("\n=== Test Evaluation ===")
        acc, rec, pre, f1 = evaluate(model, linear, test_loader, device)
        print(f"TEST Acc: {acc:.4f}, Recall: {rec:.4f}, Precision: {pre:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_real', required=True)
    parser.add_argument('--train_fake', required=True)
    parser.add_argument('--val_real', required=True)
    parser.add_argument('--val_fake', required=True)
    parser.add_argument('--test_real')
    parser.add_argument('--test_fake')
    parser.add_argument('--output_linear', required=True)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    main(args)
