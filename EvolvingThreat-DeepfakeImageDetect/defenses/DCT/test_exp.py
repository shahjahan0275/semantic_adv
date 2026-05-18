import os
import torch
import torch_dct as dct
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ----------------- Model -----------------
class DCTMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------- DCT computation -----------------
def compute_dct(image_tensor, freq_keep=128, mode="concat"):
    C, H, W = image_tensor.shape
    channel_dcts = []
    for c in range(C):
        x_tf = dct.dct_2d(image_tensor[c:c+1])
        x_tf = x_tf[:, :freq_keep, :freq_keep]
        x_tf = torch.sign(x_tf) * torch.log1p(torch.abs(x_tf))
        channel_dcts.append(x_tf)
    if mode == "mean":
        return torch.stack(channel_dcts, dim=0).mean(dim=0).flatten()
    else:
        return torch.cat(channel_dcts, dim=0).flatten()

# ----------------- Dataset -----------------
class DCTTestDataset(torch.utils.data.Dataset):
    def __init__(self, real_root, fake_root, freq_keep=128, transform=None, merge_mode="concat"):
        self.samples, self.labels = [], []
        self.freq_keep = freq_keep
        self.transform = transform
        self.merge_mode = merge_mode

        # Real images
        for p in Path(real_root).glob("*.*"):
            self.samples.append(p)
            self.labels.append(0)
        # Fake images
        for p in Path(fake_root).glob("*.*"):
            self.samples.append(p)
            self.labels.append(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        x_tf = compute_dct(img, freq_keep=self.freq_keep, mode=self.merge_mode)
        label = self.labels[idx]
        return x_tf, label

# ----------------- Metrics -----------------
def evaluate_metrics(actual, preds, probs):
    precision = precision_score(actual, preds, zero_division=0)
    recall = recall_score(actual, preds, zero_division=0)
    f1 = f1_score(actual, preds, zero_division=0)
    accuracy = accuracy_score(actual, preds)
    ap = average_precision_score(actual, probs[:, 1])
    return precision, recall, f1, accuracy, ap

# ----------------- Main -----------------
def main(args):
    transform = transforms.Compose([transforms.ToTensor()])

    # Load mean/std (per-feature)
    means = torch.load(os.path.join(args.path_to_mean_std, "means.pt")).to(DEVICE)
    stds = torch.load(os.path.join(args.path_to_mean_std, "stds.pt")).to(DEVICE)

    # --- Quick sanity check for freq_keep / merge_mode mismatch ---
    sample_img_path = next(iter(Path(args.real_root).glob("*.*")), None)
    if sample_img_path is None:
        raise RuntimeError(f"No images found in {args.real_root}")
    sample_img = transform(Image.open(sample_img_path).convert("RGB"))
    x_tf = compute_dct(sample_img, freq_keep=args.freq_keep, mode=args.merge_mode)
    input_size = x_tf.numel()
    print(f"[INFO] Inferred input size: {input_size}")

    # Safety check against training means/stds
    if means.numel() != input_size or stds.numel() != input_size:
        raise ValueError(f"[ERROR] Shape mismatch detected! "
                        f"Input DCT feature size: {input_size}, "
                        f"but loaded means/stds shapes: {means.shape}, {stds.shape}. "
                        f"This usually means freq_keep or merge_mode does not match training.")

    # Quick feature distribution check
    x_norm = (x_tf.to(DEVICE) - means) / stds
    x_min, x_max = x_norm.min().item(), x_norm.max().item()
    x_std = x_norm.std().item()
    if abs(x_min) > 5 or abs(x_max) > 5 or x_std > 2:
        print("[WARNING] Test features are far outside training distribution!")
        print(f"  min: {x_min:.3f}, max: {x_max:.3f}, std: {x_std:.3f}")
        print("  ⚠️ Likely freq_keep or merge_mode mismatch with training setup.")


    # Load model
    model = DCTMLP(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()  # Disable dropout, use running BatchNorm stats

    # Prepare dataset and dataloader
    dataset = DCTTestDataset(args.real_root, args.fake_root,
                             freq_keep=args.freq_keep,
                             transform=transform,
                             merge_mode=args.merge_mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Testing"):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # Normalize
            x_batch = (x_batch - means) / stds
            # --- DEBUG SNIPPET ---
            print("Batch shape:", x_batch.shape)
            print("Feature stats before model:")
            print("  min:", x_batch.min().item(), " max:", x_batch.max().item(), 
                " mean:", x_batch.mean().item(), " std:", x_batch.std().item())

            # Check for NaNs or Inf
            if torch.isnan(x_batch).any():
                print("[WARNING] NaNs detected in batch features!")
            if torch.isinf(x_batch).any():
                print("[WARNING] Inf detected in batch features!")
            outputs = model(x_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1, accuracy, ap = evaluate_metrics(all_labels, all_preds, all_probs)
    print(f'precision {precision:.4f}  recall {recall:.4f}  f1 {f1:.4f}  accuracy {accuracy:.4f}  AP {ap:.4f}')

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_path = os.path.join(args.save_dir, "test_results.csv")
    pd.DataFrame({
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "accuracy": [accuracy],
        "AP": [ap]
    }).to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

# ----------------- Argparse -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root", type=str, required=True)
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--path_to_mean_std", type=str, required=True)
    parser.add_argument("--freq_keep", type=int, default=128)
    parser.add_argument("--merge_mode", type=str, default="concat", choices=["concat", "mean"])
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
