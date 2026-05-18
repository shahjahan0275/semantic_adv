"""
Post-hoc Distribution Alignment (PDA) Detector
Fixed: consistent feature dimensions, output saving, and validation support
"""

import os
import joblib
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def default_image_loader(path: str, size: int = 224):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


class ImageFolderList(Dataset):
    def __init__(self, files, labels=None, loader=default_image_loader, size=224):
        self.files = files
        self.labels = labels
        self.loader = lambda p: loader(p, size=size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.loader(self.files[idx])
        y = 0 if self.labels is None else self.labels[idx]
        return x, y, self.files[idx]


def find_files(folder: str, exts=(".png", ".jpg", ".jpeg")):
    if folder is None:
        return []
    p = Path(folder)
    return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in exts])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Detector(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        if backbone_name != 'resnet50':
            raise ValueError("Only ResNet50 supported currently.")
        base = models.resnet50(pretrained=pretrained)
        in_feat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(in_feat, 1)
        self.feat_dim = in_feat

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats).squeeze(1)
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x)

# ---------------------------------------------------------------------------
# Feature post-processing
# ---------------------------------------------------------------------------

def activation_prune(feats, percentile=90.0):
    pruned = []
    for f in feats:
        c = np.percentile(f, percentile)
        pruned.append(np.minimum(f, c))
    return np.stack(pruned, axis=0)

def reduce_embeddings(feats, method='pca', out_dim=2):
    if method == 'pca':
        model = PCA(n_components=out_dim, random_state=42)
        Z = model.fit_transform(feats)
    elif method == 'tsne':
        model = TSNE(n_components=out_dim, random_state=42)
        Z = model.fit_transform(feats)
    else:
        raise ValueError("Unsupported method")
    return Z, model

# ---------------------------------------------------------------------------
# Build reference + calibration
# ---------------------------------------------------------------------------

def build_reference_set(detector, files, device, size=224, prune_p=90.0,
                        reduction='pca', batch_size=32):
    detector.eval()
    ds = ImageFolderList(files, loader=default_image_loader, size=size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)
    feats = []
    paths = []
    with torch.no_grad():
        for xb, _, bpaths in dl:
            xb = xb.to(device)
            f = detector.extract_features(xb).cpu().numpy()
            feats.append(f)
            paths.extend(bpaths)
    feats = np.concatenate(feats, axis=0)
    pruned = activation_prune(feats, prune_p)
    Z, reducer = reduce_embeddings(pruned, method=reduction)
    return Z, paths, pruned, reducer

def calibrate_threshold(Z_regenerated, Z_reference, k=5, percentile=95.0):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Z_reference)
    distances, _ = nbrs.kneighbors(Z_regenerated)
    dk = distances[:, k]
    tau = np.percentile(dk, percentile)
    return float(tau)

# ---------------------------------------------------------------------------
# Train detector
# ---------------------------------------------------------------------------

def train_detector(train_real_files, train_fake_files, device, save_path,
                   epochs=10, batch_size=32, lr=1e-4, size=224,
                   val_real_files=None, val_fake_files=None):
    detector = Detector().to(device)
    train_files = train_real_files + train_fake_files
    train_labels = [0]*len(train_real_files) + [1]*len(train_fake_files)
    train_dl = DataLoader(ImageFolderList(train_files, labels=train_labels, size=size),
                          batch_size=batch_size, shuffle=True, num_workers=4)

    if val_real_files and val_fake_files:
        val_files = val_real_files + val_fake_files
        val_labels = [0]*len(val_real_files) + [1]*len(val_fake_files)
        val_dl = DataLoader(ImageFolderList(val_files, labels=val_labels, size=size),
                            batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_dl = None

    optim_ = optim.Adam(detector.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        detector.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.float().to(device)
            logits = detector(xb)
            loss = loss_fn(logits, yb)
            optim_.zero_grad()
            loss.backward()
            optim_.step()

            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        print(f"Epoch {ep+1}/{epochs} | Train Loss: {total_loss/total:.4f} | Acc: {100*correct/total:.2f}%", end='')

        if val_dl:
            detector.eval()
            val_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for xb, yb, _ in val_dl:
                    xb, yb = xb.to(device), yb.float().to(device)
                    out = detector(xb)
                    loss = loss_fn(out, yb)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    v_correct += (preds == yb).sum().item()
                    val_loss += loss.item() * xb.size(0)
                    v_total += xb.size(0)
            print(f" | Val Loss: {val_loss/v_total:.4f} | Val Acc: {100*v_correct/v_total:.2f}%")
        else:
            print()

    os.makedirs(Path(save_path).parent, exist_ok=True)
    torch.save(detector.state_dict(), save_path)
    print(f"✅ Saved model to {save_path}")
    return detector

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-real", type=str)
    parser.add_argument("--train-fake", type=str)
    parser.add_argument("--ref-fake", type=str)
    parser.add_argument("--calib-real", type=str)
    parser.add_argument("--val-real", type=str, default=None)
    parser.add_argument("--val-fake", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-detector", type=str, default="checkpoints/detector.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--reduction", type=str, default="pca", choices=["pca", "tsne"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_detector).parent
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Train detector ----------------
    if args.train_real and args.train_fake:
        real_files = find_files(args.train_real)
        fake_files = find_files(args.train_fake)
        print(f"Training on {len(real_files)} real / {len(fake_files)} fake samples...")
        val_real = find_files(args.val_real) if args.val_real else None
        val_fake = find_files(args.val_fake) if args.val_fake else None
        detector = train_detector(real_files, fake_files, device, args.save_detector,
                                  epochs=args.epochs, size=args.size,
                                  val_real_files=val_real, val_fake_files=val_fake)
    else:
        detector = Detector().to(device)
        detector.load_state_dict(torch.load(args.save_detector, map_location=device))
        print(f"Loaded existing detector from {args.save_detector}")

    # ---------------- Build reference set ----------------
    if args.ref_fake:
        ref_files = find_files(args.ref_fake)
        Z_ref, ref_paths, ref_pruned, reducer = build_reference_set(detector, ref_files,
                                                                    device, size=args.size,
                                                                    reduction=args.reduction)
        np.savez_compressed(save_dir / "reference_set.npz",
                            Z=Z_ref, paths=ref_paths, pruned=ref_pruned)
        
        # ✅ Save PCA or t-SNE reducer properly
        if args.reduction == "pca":
            joblib.dump(reducer, save_dir / "pca_model.pkl")
            print(f"✅ Saved PCA model as {save_dir}/pca_model.pkl")
        else:
            joblib.dump(reducer, save_dir / "tsne_model.pkl")
            print(f"✅ Saved TSNE model as {save_dir}/tsne_model.pkl")
        
        print(f"✅ Saved reference_set.npz and reducer in {save_dir}")
    else:
        ref_data = np.load(save_dir / "reference_set.npz", allow_pickle=True)
        Z_ref, ref_pruned = ref_data["Z"], ref_data["pruned"]
        if (save_dir / "pca_model.pkl").exists():
            reducer = joblib.load(save_dir / "pca_model.pkl")
        elif (save_dir / "tsne_model.pkl").exists():
            reducer = joblib.load(save_dir / "tsne_model.pkl")
        else:
            raise FileNotFoundError("No reducer model (.pkl) found.")


    # ---------------- Calibrate threshold ----------------
    if args.calib_real:
        calib_files = find_files(args.calib_real)
        feats = []
        with torch.no_grad():
            for p in calib_files:
                x = default_image_loader(p, size=args.size).unsqueeze(0).to(device)
                f = detector.extract_features(x).cpu().numpy()
                feats.append(f)
        feats = np.concatenate(feats, axis=0)
        feats_pruned = activation_prune(feats)
        Z_calib = reducer.transform(feats_pruned)
        tau = calibrate_threshold(Z_calib, Z_ref)
        np.savez_compressed(save_dir / "calibration.npz", tau=tau)
        print(f"✅ Saved calibration.npz (tau={tau:.4f})")
    else:
        tau = float(np.load(save_dir / "calibration.npz")["tau"])
        print(f"Loaded tau = {tau:.4f}")

    print("🎯 PDA training and calibration complete.")
