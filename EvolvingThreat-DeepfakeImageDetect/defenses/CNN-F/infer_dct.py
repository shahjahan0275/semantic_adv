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
    f1_score, average_precision_score
)

from networks.trainer_dct_old import Trainer
from data.datasets import extract_dct_stats, make_patches


# ============================================================
# EXACT SAME PATCH + DCT PIPELINE AS TRAINING
# ============================================================
def build_patches(img, K=16):
    """
    Extract exactly K random 224x224 patches (same as training).
    No resizing. No cropping. No fixed grid.
    """
    all_patches = make_patches(img, patch=224, stride=224)

    # sample EXACTLY K patches (same rule as datasets.py)
    if len(all_patches) > K:
        idx = torch.randperm(len(all_patches))[:K]
        all_patches = [all_patches[i] for i in idx]

    # If image is small and returns fewer, pad by repeating last patch
    while len(all_patches) < K:
        all_patches.append(all_patches[-1])

    # convert to normalized tensors
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    t_list = []
    for p in all_patches:
        t = norm(to_tensor(p))
        t_list.append(t)

    return torch.stack(t_list)      # [K,3,224,224]


def dct_transform(img):
    """
    Produce 81920-dim DCT vector EXACTLY like training.
    """
    img_tensor = transforms.ToTensor()(img).float()
    dct_feat = extract_dct_stats(img_tensor)
    return dct_feat


# ============================================================
# CMD ARGS
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
# Dummy opt for Trainer
# ============================================================
class DummyOpt:
    checkpoints_dir = "./"
    name = "infer"
    gpu_ids = [0]
    isTrain = False
    continue_train = False
    new_optim = False

    lr = 1e-4
    beta1 = 0.5
    optim = "adam"

    epoch = "latest"
    modeltype = "cnn"

    num_patches = opt.num_patches


dummy_opt = DummyOpt()


# ============================================================
# Load Trainer
# ============================================================
model = Trainer(dummy_opt)
model.to(device)
model.eval()

print(f"\nLoading checkpoint: {opt.model_path}")
ckpt = torch.load(opt.model_path, map_location=device)

model.model_cnn.load_state_dict(ckpt["model_cnn"])
model.model_mlp.load_state_dict(ckpt["model_mlp"])
model.final_fc.load_state_dict(ckpt["final_fc"])
model.eval()


# ============================================================
# Dataset for inference
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

        # CNN: random K patches
        patches = build_patches(img, K=self.K)     # [K,3,224,224]

        # DCT: 81920 dims
        dct_feat = dct_transform(img)

        return patches, dct_feat, label


# ============================================================
# Load data
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
# Inference
# ============================================================
y_true, y_pred = [], []

with torch.no_grad():
    for loader in all_loaders:
        for patches, dct, label in tqdm(loader):

            patches = patches.to(device)      # [B,K,3,224,224]
            dct = dct.to(device)              # [B,81920]
            label = label.to(device).float()  # [B]

            # Run through trainer forward
            logits = model.forward(patches, dct)  # [B,1]

            probs = torch.sigmoid(logits).flatten().cpu().numpy()

            y_pred.extend(probs.tolist())
            y_true.extend(label.cpu().numpy().tolist())


# ============================================================
# Metrics
# ============================================================
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_bin  = (y_pred > 0.5).astype(int)

acc  = accuracy_score(y_true, y_bin)
prec = precision_score(y_true, y_bin)
rec  = recall_score(y_true, y_bin)
f1   = f1_score(y_true, y_bin)
ap   = average_precision_score(y_true, y_pred)

print("\n============== RESULTS ==============")
print(f"Accuracy                : {acc:.4f}")
print(f"Precision               : {prec:.4f}")
print(f"Recall                  : {rec:.4f}")
print(f"F1 score                : {f1:.4f}")
print(f"Average Precision (AP)  : {ap:.4f}")
print("====================================\n")
