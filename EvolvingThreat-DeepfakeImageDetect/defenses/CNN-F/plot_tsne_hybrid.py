import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Use your new Trainer and dataset logic
from networks.trainer_dct import Trainer
from data.datasets import extract_dct_stats, make_patches

# ============================================================
# 1. SETUP PATHS & HYPERPARAMETERS
# ============================================================
data_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/CLIPResNet/"
#data_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/EfficientNet/"
#data_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/data/AdvImages_w_SurrogateModels/ViT/"
model_path = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/DCT_ResNet50_LM12/model_epoch_latest.pth"

batch_size = 16 
num_patches = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. UPDATED HELPER FUNCTIONS (Matching final test code)
# ============================================================
def build_patches_local(img, K=16):
    patches = make_patches(img, patch=224)
    # Match final test code: take first K, then pad
    patches = patches[:K]
    while len(patches) < K:
        patches.append(patches[-1])
        
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.stack([norm(to_tensor(p)) for p in patches])

def dct_transform_local(img, mean=None, std=None):
    img_tensor = transforms.ToTensor()(img).float()
    feat = extract_dct_stats(img_tensor)
    # NEW: Apply normalization logic from your final test code
    if mean is not None and std is not None:
        feat = (feat - mean) / (std + 1e-6)
    return feat

# ============================================================
# 3. DUMMY OPT & MODEL LOADING
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
# Prevent auto load of pretrained weights to avoid errors
model.load_networks = lambda *a, **kw: None
model.to(device)

print(f"Loading hybrid checkpoint: {model_path}")
ckpt = torch.load(model_path, map_location=device)

# LOAD DCT STATS FROM CHECKPOINT
dct_mean = ckpt.get("dct_mean", None)
dct_std = ckpt.get("dct_std", None)
print(f"DCT Stats Found: {dct_mean is not None}")

model.model_cnn.load_state_dict(ckpt["model_cnn"])
model.model_mlp.load_state_dict(ckpt["model_mlp"])
model.final_fc.load_state_dict(ckpt["final_fc"])
model.eval()

# Move stats to CPU for the DataLoader workers
dct_mean_cpu = dct_mean.cpu() if dct_mean is not None else None
dct_std_cpu = dct_std.cpu() if dct_std is not None else None

# ============================================================
# 4. HYBRID FEATURE EXTRACTION LOGIC
# ============================================================
def extract_fused_features(img_batch, dct_batch):
    B, P, C, H, W = img_batch.shape
    img_flat = img_batch.reshape(B * P, C, H, W)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # CNN path
            flat = model.model_cnn(img_flat) 
            cnn_feat = torch.stack([s.mean(0) for s in torch.split(flat, P, dim=0)], dim=0)
            # DCT path (MLP expects normalized input)
            dct_latent = model.model_mlp(dct_batch)
            # Joint Latent Space
            fused = torch.cat([cnn_feat, dct_latent], dim=1)
    return fused

# ============================================================
# 5. DATA LOADER
# ============================================================
class TSNEHybridDataset(ImageFolder):
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = Image.open(path).convert("RGB")
        patches = build_patches_local(img, K=num_patches)
        # Use CPU stats for normalization in worker process
        dct_feat = dct_transform_local(img, mean=dct_mean_cpu, std=dct_std_cpu)
        return patches, dct_feat, label

dataset = TSNEHybridDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ============================================================
# 6. EXECUTE EXTRACTION
# ============================================================
features = []
labels = []

print("Extracting fused CNN+DCT features...")
for patches, dct, target in tqdm(loader):
    patches, dct = patches.to(device), dct.to(device)
    fused_feat = extract_fused_features(patches, dct)
    features.append(fused_feat.cpu().numpy())
    labels.append(target.numpy())

features = np.concatenate(features, axis=0).astype('float64')
labels = np.concatenate(labels, axis=0)

# ============================================================
# 7. T-SNE & PLOTTING
# ============================================================
print(f"Running t-SNE on {len(features)} samples...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200.0, init='pca', random_state=42, n_jobs=-1)
tsne_results = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
# Use distinctive colors to highlight separation
#colors = ['#1f77b4', '#d62728'] 
colors = ['green', 'red']
for i, color, name in zip([0, 1], colors, ['Real', 'Fake']):
    idx = np.where(labels == i)
    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                c=color, label=name, alpha=0.6, s=25, edgecolors='w', linewidth=0.5)

plt.legend()
plt.title(f't-SNE Hybrid: CNN + Normalized 4th-order DCT Features\nDataset: {os.path.basename(data_dir.strip("/"))}')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('tsne_CNN-F_modified_ViT.png', dpi=300)
print("Saved:tsne_CNN-F_modified_ViT.png")
