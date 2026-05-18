import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, precision_recall_curve
)

from networks.trainer_dct import Trainer
from data.datasets import extract_dct_stats, make_patches
from data import create_dataloader
from options.train_options import TrainOptions
import sys

# ============================================================
# Patch Builder
# ============================================================
def build_patches_infer(img, K=16):
    patches = make_patches(img, patch=224)

    # Keep exactly first K patches
    patches = patches[:K]

    # Pad if fewer than K
    while len(patches) < K:
        patches.append(patches[-1])

    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tensor_list = [norm(to_tensor(p)) for p in patches]
    return torch.stack(tensor_list)


# ============================================================
# DCT (with optional normalization)
# ============================================================
def dct_transform_infer(img, mean=None, std=None):
    img_t = transforms.ToTensor()(img)
    feat = extract_dct_stats(img_t)

    if mean is not None and std is not None:
        feat = (feat - mean) / (std + 1e-6)

    return feat


# ============================================================
# CMD args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dir", nargs="+", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--num_patches", type=int, default=16)
opt = parser.parse_args()

device = torch.device("cpu" if opt.use_cpu else "cuda")
print(f"Using device: {device}")


# ============================================================
# Dummy opt for Trainer (Inference Only)
# ============================================================
class DummyOpt:
    checkpoints_dir = "./"
    name = "infer"
    gpu_ids = [0] if device.type == "cuda" else []
    isTrain = False
    continue_train = False
    new_optim = False
    lr = 1e-4
    beta1 = 0.5
    optim = "adam"

    # Disable pretrained loading
    epoch = None
    modeltype = None

    num_patches = opt.num_patches


dummy_opt = DummyOpt()

# ============================================================
# Load Trainer (disable internal network loading)
# ============================================================
model = Trainer(dummy_opt)

# Disable auto load of pretrained weights
model.load_networks = lambda *a, **kw: None

model.to(device)
model.eval()

print(f"\nLoading checkpoint: {opt.model_path}")
ckpt = torch.load(opt.model_path, map_location=device)


# ============================================================
# Auto-fix checkpoint if DCT stats missing
# ============================================================
if ("dct_mean" not in ckpt) or ("dct_std" not in ckpt):
    print("\n[WARNING] Checkpoint missing DCT mean/std — rebuilding them now...")

    # Prevent TrainOptions from parsing inference args
    _saved_argv = sys.argv
    sys.argv = ["infer"]
    fix_opt = TrainOptions().parse()
    sys.argv = _saved_argv

    # Set your real training path
    fix_opt.dataroot = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/mydataset"

    loader_fix = create_dataloader(fix_opt)
    ds_fix = loader_fix.dataset

    # --------------------------------------------------------
    # Correct unwrapping of ConcatDataset → BinaryHybridDataset
    # --------------------------------------------------------
    if isinstance(ds_fix, torch.utils.data.ConcatDataset):
        # assume first dataset is REAL dataset
        real_ds = ds_fix.datasets[0]
        print("[INFO] Found ConcatDataset → using REAL dataset:", type(real_ds))
    else:
        real_ds = ds_fix
        print("[INFO] Using dataset:", type(real_ds))

    print("[INFO] Scanning REAL dataset to build DCT mean/std...")
    for i in range(len(real_ds)):
        _ = real_ds[i]      # triggers internal accumulation

    # Now finalize ONLY this dataset
    if hasattr(real_ds, "finalize_stats"):
        real_ds.finalize_stats()
    else:
        raise RuntimeError("ERROR: real_ds has no finalize_stats() — expected BinaryHybridDataset")

    print("[INFO] Scan complete.")

    # Fetch accumulated stats
    dct_mean_fix = real_ds.get_dct_mean().cpu()
    dct_std_fix  = real_ds.get_dct_std().cpu()


    print("[INFO] Injecting DCT mean/std into checkpoint...")

    ckpt["dct_mean"] = dct_mean_fix
    ckpt["dct_std"] = dct_std_fix

    fixed_path = opt.model_path.replace(".pth", "_fixed.pth")
    torch.save(ckpt, fixed_path)
    print(f"[INFO] Saved fixed checkpoint → {fixed_path}")

    dct_mean = dct_mean_fix.to(device)
    dct_std  = dct_std_fix.to(device)
else:
    print("[INFO] DCT mean/std found in checkpoint.")
    dct_mean = ckpt["dct_mean"].to(device)
    dct_std  = ckpt["dct_std"].to(device)


print("DCT mean/std loaded:", dct_mean is not None, dct_std is not None)

# ============================================================
# Auto-fix checkpoint if DCT stats missing finish
# ============================================================

model.model_cnn.load_state_dict(ckpt["model_cnn"])
model.model_mlp.load_state_dict(ckpt["model_mlp"])
model.final_fc.load_state_dict(ckpt["final_fc"])

model.eval()

# ---- Load DCT statistics ----
dct_mean = ckpt.get("dct_mean", None)
dct_std  = ckpt.get("dct_std", None)

if dct_mean is not None:
    dct_mean = dct_mean.to(device)
if dct_std is not None:
    dct_std = dct_std.to(device)

print("DCT mean/std loaded:", dct_mean is not None, dct_std is not None)

# ---- CPU copies for DataLoader workers ----
dct_mean_cpu = dct_mean.cpu() if dct_mean is not None else None
dct_std_cpu = dct_std.cpu() if dct_std is not None else None


# ============================================================
# Hybrid Dataset
# ============================================================
class HybridInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, K=16):
        from torchvision.datasets import ImageFolder
        self.ds = ImageFolder(rootdir)
        self.classes = self.ds.classes
        self.K = K

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, label = self.ds.imgs[idx]
        img = Image.open(path).convert("RGB")

        patches = build_patches_infer(img, self.K)
        #dct_feat = dct_transform_infer(img, mean=dct_mean, std=dct_std)
        dct_feat = dct_transform_infer(img, mean=dct_mean_cpu, std=dct_std_cpu)

        return patches, dct_feat, label


# ============================================================
# Create loaders for each directory
# ============================================================
all_loaders = []
for d in opt.dir:
    ds = HybridInferenceDataset(d, K=opt.num_patches)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers
    )
    all_loaders.append(loader)


# ============================================================
# Inference Loop
# ============================================================
y_true, y_pred = [], []

with torch.no_grad():
    for loader in all_loaders:
        for patches, dct, label in tqdm(loader):
            patches = patches.to(device)
            dct = dct.to(device)
            label = label.float().unsqueeze(1).to(device)

            logits = model.forward(patches, dct)
            probs = torch.sigmoid(logits).flatten().cpu().numpy()

            y_pred.extend(probs.tolist())
            y_true.extend(label.cpu().numpy().flatten().tolist())


# ============================================================
# Metrics + optimal threshold
# ============================================================
'''
y_true = np.array(y_true)
y_pred = np.array(y_pred)

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print("\nBest threshold =", best_threshold)


y_bin = (y_pred >= best_threshold).astype(int)

acc = accuracy_score(y_true, y_bin)
prec = precision_score(y_true, y_bin)
rec = recall_score(y_true, y_bin)
f1 = f1_score(y_true, y_bin)
ap = average_precision_score(y_true, y_pred)

print("\n============== RESULTS ==============")
print(f"Accuracy                : {acc:.4f}")
print(f"Precision               : {prec:.4f}")
print(f"Recall                  : {rec:.4f}")
print(f"F1 score                : {f1:.4f}")
print(f"Average Precision (AP)  : {ap:.4f}")
print("====================================\n")
'''
# ============================================================
# Metrics (Fixed Threshold for Adversarial Evaluation)
# ============================================================
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# In adversarial evaluation, the threshold must be fixed to what the 
# detector would use in the real world (typically 0.5) to measure true robustness.
evaluation_threshold = 0.50

print(f"\nEvaluating at strict threshold = {evaluation_threshold}")

y_bin = (y_pred >= evaluation_threshold).astype(int)

acc = accuracy_score(y_true, y_bin)
prec = precision_score(y_true, y_bin, zero_division=0)
rec = recall_score(y_true, y_bin)
f1 = f1_score(y_true, y_bin)
ap = average_precision_score(y_true, y_pred)

print("\n============== RESULTS ==============")
print(f"Accuracy                : {acc:.4f}")
print(f"Precision               : {prec:.4f}")
print(f"Recall                  : {rec:.4f}")
print(f"F1 score                : {f1:.4f}")
print(f"Average Precision (AP)  : {ap:.4f}")
print("====================================\n")