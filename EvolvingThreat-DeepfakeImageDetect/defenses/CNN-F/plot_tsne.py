import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Import your resnet50 correctly
from networks.resnet import resnet50 

# 1. Setup Data Loading - REMOVED CROP/RESIZE
data_dir = '/media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/Test_data_Adv/AdvImages_w_SurrogateModels/CLIPResNet' 
model_path = '/media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/weights/blur_jpg_prob0.1.pth'
#model_path = '/media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/CNN-F/checkpoints/use_a_name/model_epoch_best.pth'

# IMPORTANT: Batch size 4 for 1024x1024 images on 20GB GPU
batch_size = 4 

trans = transforms.Compose([
    transforms.ToTensor(), # Full 1024x1024 resolution
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(data_dir, transform=trans)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 2. Load Model and Modify for Feature Extraction
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.cuda()
model.eval()

# Extracting the feature extractor (all layers except the final FC)
# ResNet50 structure: ... -> GlobalAvgPool -> Flatten -> FC
# By taking everything but the last child, we get the output of the GAP layer.
feature_extractor = nn.Sequential(*list(model.children())[:-1])

'''
# ============================================================
# 2. Load Model and Modify for Feature Extraction
# ============================================================
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')

# Check if it's the hybrid checkpoint (DCT style) or original style
if 'model_cnn' in state_dict:
    print("Detected Hybrid/DCT checkpoint. Loading model_cnn branch...")
    model.load_state_dict(state_dict['model_cnn'], strict=False)
elif 'model' in state_dict:
    print("Detected Standard checkpoint. Loading model branch...")
    model.load_state_dict(state_dict['model'])
else:
    # If the state_dict is just the weights themselves without a wrapper key
    print("Detected raw state_dict. Loading directly...")
    model.load_state_dict(state_dict)

model.cuda()
model.eval()

# Extracting the feature extractor (all layers except the final FC)
# This gets the 2048-dimensional feature vector before the final classification
feature_extractor = nn.Sequential(*list(model.children())[:-1])
'''

# 3. Extract Features
features = []
labels = []

print(f"Extracting features at full resolution (1024x1024) with batch size {batch_size}...")
with torch.no_grad():
    for data, target in tqdm(loader):
        data = data.cuda()
        feat = feature_extractor(data)
        feat = feat.view(feat.size(0), -1)
        features.append(feat.cpu().numpy())
        labels.append(target.numpy())

# FIX 1: Explicitly cast to float64 to prevent dtype mismatch errors in sklearn
features = np.concatenate(features, axis=0).astype('float64')
labels = np.concatenate(labels, axis=0)

# 4. Perform t-SNE
print("Running t-SNE on 2000 samples...")
# FIX 2: Change 'auto' to a numeric value (200.0) 
# FIX 3: Ensure init is 'random' if 'pca' continues to throw errors in your specific version
tsne = TSNE(
    n_components=2, 
    perplexity=40, 
    learning_rate=200.0, # Numeric value fixes the 'UFuncTypeError'
    init='pca',          # If this still errors, change to 'random'
    random_state=42, 
    n_jobs=-1 
)
tsne_results = tsne.fit_transform(features)


# 5. Plotting
plt.figure(figsize=(12, 8))
colors = ['green', 'red'] # Professional Blue and Red
target_names = ['Real', 'Fake']

for i, color, label_name in zip([0, 1], colors, target_names):
    indices = np.where(labels == i)
    plt.scatter(tsne_results[indices, 0], 
                tsne_results[indices, 1], 
                c=color, label=label_name, alpha=0.6, edgecolors='w', linewidth=0.5, s=25)

plt.legend()
#plt.title('t-SNE:Original model Full Resolution (1024x1024) Features (CNN-F)\nDataset: {os.path.basename(data_dir.strip("/"))}')
plt.title(f't-SNE Original: Full Resolution Features CNN-F\nDataset: {os.path.basename(data_dir.strip("/"))}')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('tsne_original_CLIPResNet_CNN-F.png', dpi=300) # High-res save for papers
print("Plot saved as tsne_original_CLIPResNet_CNN-F.png")
plt.show()