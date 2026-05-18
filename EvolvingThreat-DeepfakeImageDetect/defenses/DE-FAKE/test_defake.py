import torch
import clip
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score
)
import argparse


# ============================================================
# Dataset (image-only, same as training)
# ============================================================
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

        return image, self.label_value, img_path


# ============================================================
# Classifier (must match training)
# ============================================================
class Classifier(nn.Module):
    def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============================================================
# TEST FUNCTION
# ============================================================
def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP ViT-B/32
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Same normalization & resizing as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Datasets
    test_real = CSVDataset(args.test_real, transform, label_value=0)
    test_fake = CSVDataset(args.test_fake, transform, label_value=1)

    test_dataset = torch.utils.data.ConcatDataset([test_real, test_fake])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load your trained classifier
    classifier = Classifier(input_dim=512).to(device)
    classifier.load_state_dict(torch.load(args.output_linear, map_location=device))
    classifier.eval()

    y_true, y_pred = [], []
    rows = []   # <--- store CSV rows

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            # CLIP image embedding only
            img_emb = clip_model.encode_image(images)  # shape [B,512]

            out = classifier(img_emb.float())
            pred = out.argmax(1)

            # store results
            for p, t, ip in zip(pred.cpu().numpy(), labels.cpu().numpy(), paths):
                rows.append({
                    "imagepath": ip,
                    "true_label": int(t),
                    "predicted_label": int(p)
                })

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # =====================
    # Save predictions CSV + metrics
    # =====================
    df_out = pd.DataFrame(rows)

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    # Append metrics as extra rows
    metrics_rows = pd.DataFrame([
        {"imagepath": "METRIC", "true_label": "Precision", "predicted_label": precision},
        {"imagepath": "METRIC", "true_label": "Recall",    "predicted_label": recall},
        {"imagepath": "METRIC", "true_label": "F1",        "predicted_label": f1},
        {"imagepath": "METRIC", "true_label": "Accuracy",  "predicted_label": acc},
    ])

    # Combine and save
    df_final = pd.concat([df_out, metrics_rows], ignore_index=True)
    df_final.to_csv(args.save_csv, index=False)

    print(f"\nSaved prediction + metrics CSV: {args.save_csv}\n")



    # =====================
    # Final Metrics
    # =====================
    print("\n============== FINAL TEST METRICS ==============")
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:   ", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_true, y_pred, average='weighted'))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("================================================\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_real", type=str, required=True)
    parser.add_argument("--test_fake", type=str, required=True)
    parser.add_argument("--output_linear", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)

    args = parser.parse_args()
    test(args)
