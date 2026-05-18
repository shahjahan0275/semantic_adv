import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Your project imports
from networks.trainer_dct import Trainer
from data.datasets import extract_dct_stats, make_patches
from data import create_dataloader
from options.train_options import TrainOptions

# ============================================================
# 1. SETUP PATHS & HYPERPARAMETERS
# ============================================================
data_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/"
#data_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/"
model_path = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/DCT_ResNet50_LM12/model_epoch_latest.pth"
train_dataset_path = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/mydataset"

batch_size = 16
num_patches = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def build_patches_local(img, K=16):
    patches = make_patches(img, patch=224)
    patches = patches[:K]
    while len(patches) < K:
        patches.append(patches[-1])
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.stack([norm(to_tensor(p)) for p in patches])

def dct_transform_local(img, mean=None, std=None):
    img_tensor = transforms.ToTensor()(img).float()
    feat = extract_dct_stats(img_tensor)
    if mean is not None and std is not None:
        feat = (feat - mean) / (std + 1e-6)
    return feat

# ============================================================
# 3. MODEL LOADING & AUTO-FIX DCT STATS
# ============================================================
class DummyOpt:
    checkpoints_dir = "./checkpoints"
    name = "tsne_inference"
    gpu_ids = [0] if torch.cuda.is_available() else []
    isTrain = False
    continue_train = False
    num_patches = num_patches
    modeltype = None
    epoch = None
    init_gain = 0.02
    optim = "adam"
    lr = 0.0001
    beta1 = 0.5

opt = DummyOpt()
model = Trainer(opt)
model.load_networks = lambda *a, **kw: None # Disable auto-load
model.to(device)

print(f"Loading hybrid checkpoint: {model_path}")
ckpt = torch.load(model_path, map_location=device)

# --- AUTO-FIX LOGIC ---
if ("dct_mean" not in ckpt) or ("dct_std" not in ckpt):
    print("\n[WARNING] Checkpoint missing DCT stats — Rebuilding from training data...")
    _saved_argv = sys.argv
    sys.argv = ["tsne_fix"] # Dummy arg to bypass TrainOptions parser
    fix_opt = TrainOptions().parse()
    sys.argv = _saved_argv
    
    fix_opt.dataroot = train_dataset_path
    loader_fix = create_dataloader(fix_opt)
    ds_fix = loader_fix.dataset

    # Unwrap ConcatDataset if necessary
    real_ds = ds_fix.datasets[0] if isinstance(ds_fix, torch.utils.data.ConcatDataset) else ds_fix
    
    print(f"[INFO] Scanning REAL training dataset at: {train_dataset_path}")
    for i in tqdm(range(len(real_ds)), desc="Calculating Stats"):
        _ = real_ds[i] # Triggers internal accumulation in BinaryHybridDataset
        
    if hasattr(real_ds, "finalize_stats"):
        real_ds.finalize_stats()
        dct_mean = real_ds.get_dct_mean().to(device)
        dct_std = real_ds.get_dct_std().to(device)
    else:
        raise RuntimeError("Dataset does not support finalize_stats(). Check your data class.")
else:
    print("[INFO] DCT mean/std found in checkpoint.")
    dct_mean = ckpt["dct_mean"].to(device)
    dct_std = ckpt["dct_std"].to(device)

# Load weights into model
model.model_cnn.load_state_dict(ckpt["model_cnn"])
model.model_mlp.load_state_dict(ckpt["model_mlp"])
model.final_fc.load_state_dict(ckpt["final_fc"])
model.eval()

dct_mean_cpu = dct_mean.cpu()
dct_std_cpu = dct_std.cpu()

# ============================================================
# 4. FEATURE EXTRACTION & DATA LOADER
# ============================================================
def extract_fused_features(img_batch, dct_batch):
    B, P, C, H, W = img_batch.shape
    img_flat = img_batch.reshape(B * P, C, H, W)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # CNN path -> global average across patches
            flat = model.model_cnn(img_flat)
            cnn_feat = torch.stack([s.mean(0) for s in torch.split(flat, P, dim=0)], dim=0)
            # DCT path -> latent vector
            dct_latent = model.model_mlp(dct_batch)
            # Fused vector for t-SNE
            fused = torch.cat([cnn_feat, dct_latent], dim=1)
    return fused

class TSNEHybridDataset(ImageFolder):
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = Image.open(path).convert("RGB")
        patches = build_patches_local(img, K=num_patches)
        dct_feat = dct_transform_local(img, mean=dct_mean_cpu, std=dct_std_cpu)
        return patches, dct_feat, label

dataset = TSNEHybridDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ============================================================
# 5. EXECUTION & PLOTTING
# ============================================================
features, labels = [], []
print("Extracting fused features...")
for patches, dct, target in tqdm(loader):
    patches, dct = patches.to(device), dct.to(device)
    fused_feat = extract_fused_features(patches, dct)
    features.append(fused_feat.cpu().numpy())
    labels.append(target.numpy())

features = np.concatenate(features, axis=0).astype('float64')
labels = np.concatenate(labels, axis=0)

print(f"Running t-SNE on {len(features)} samples...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200.0, init='pca', random_state=42, n_jobs=-1)
tsne_results = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
for i, color, name in zip([0, 1], ['green', 'red'], ['Real', 'Fake']):
    idx = np.where(labels == i)
    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=color, label=name, alpha=0.6, s=25, edgecolors='w', linewidth=0.5)

plt.legend()
plt.title(f't-SNE Hybrid (ViT): CNN + Normalized DCT Stats\nDataset: {os.path.basename(data_dir.strip("/"))}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('tsne_CNN-F_modified_CLIPResNet.png', dpi=300)
print("Saved: tsne_CNN-F_modified_CLIPResNet.png")
