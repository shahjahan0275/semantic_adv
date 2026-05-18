from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
import torch_dct as dct
import torch.nn.functional as F
from models.transformer_attention import TransformerAttention
import torchvision.transforms as transforms
from .clip.model import VisionTransformer
from .mlp import MLP

from .denoiser import run_denoiser       # wrapper for test.py logic (Step 2)    3 branch
from .noise_extractor import run_extractnoise, NoiseProjector  # wrapper for extractnoise.py logic (Step 3)   3 branch

# Global noise projector instance (will be initialized dynamically in forward)
_NOISE_PROJECTOR = None

# -------------------------------
# Channel dimensions for different CLIP backbones
# -------------------------------

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14-penultimate" : 1024
}

# -------------------------------
# Normalization constants
# -------------------------------

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

# FOR 1D DCT Transform
def apply_dct(feat: torch.Tensor) -> torch.Tensor:
    """
    Apply 1D DCT-II on the last dimension of the feature tensor.
    Args:
        feat: (B, D) tensor
    Returns:
        DCT-transformed tensor of same shape
    """
    return dct.dct(feat, norm='ortho')

# FOR 2D DCT Transform
def apply_2d_dct(feat: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D DCT-II on each feature vector (B, D) by reshaping to square.
    Args:
        feat: Tensor of shape (B, D)
    Returns:
        Tensor of shape (B, D) after 2D DCT
    """
    B, D = feat.shape
    side = int(D ** 0.5)
    assert side * side == D, f"Feature dim {D} is not a perfect square for 2D DCT"

    # Reshape to (B, H, W)
    feat_2d = feat.view(B, side, side)

    # Apply 1D DCT along rows
    feat_dct = dct.dct(feat_2d, norm='ortho', dim=1)
    # Apply 1D DCT along cols
    feat_dct = dct.dct(feat_dct, norm='ortho', dim=2)

    # Flatten back
    return feat_dct.view(B, D)


# =========================================================
# CLIP:ViT-L/14 Feature Extractor (Visual Backbone)
# =========================================================

class CLIPModel(nn.Module):
    """UFD"""
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
        # --- REPLACEMENT ---
        #self.fc = MLP(
            #input_dim=CHANNELS[name],
            #hidden_dims=[512, 256],
            #output_dim=num_classes
        #)


    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)    # CLIP features
        if return_feature:
            return features
        return self.fc(features)                 # classification layer/robust classifier

# =========================================================
# Penultimate Layer Feature Extractor (Paper: CLIP*)
# =========================================================

class CLIPModelPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModelPenultimateLayer, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.register_hook()
        self.fc = nn.Linear(CHANNELS[name+"-penultimate"], num_classes)    # classification layer
        # --- REPLACEMENT ---
        #self.fc = MLP(
            #input_dim=CHANNELS[name+"-penultimate"],
            #hidden_dims=[512, 256],
            #output_dim=num_classes
        #)



    def register_hook(self):
        
        def hook(module, input, output):
            self.features = torch.clone(output)    # penultimate features
        for name, module in self.model.visual.named_children():
            if name == "ln_post":                  # hook on penultimate layer
                module.register_forward_hook(hook)
        return 

    def forward(self, x):
        self.model.encode_image(x)                  # run CLIP forward
        return self.fc(self.features)               # classification from penultimate features

# =========================================================
# Patch Shuffle + Self-Attention Invariance Extraction
# =========================================================

class CLIPModelShuffleAttentionPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1,shuffle_times=1, patch_size=32, original_times=1):
        super(CLIPModelShuffleAttentionPenultimateLayer, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.shuffle_times = shuffle_times
        self.original_times = original_times
        #self.patch_size = patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        if self.patch_size <= 1:
            print(f"[clip_models] Warning: invalid patch_size={self.patch_size}, resetting to 14")
            self.patch_size = 14

        # CLIP:ViT-L/14 backbone (Visual Backbone)
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        self.register_hook()

        # Self-Attention Invariance Extraction Layer
        #self.attention_head = TransformerAttention(CHANNELS[name+"-penultimate"], shuffle_times + original_times, last_dim=num_classes)
        # Include +1 for the noise branch
        num_branches = self.shuffle_times + self.original_times + 1
        self.attention_head = TransformerAttention(
            input_dim=CHANNELS[self.name+"-penultimate"],
            num_branches=num_branches,
            last_dim=num_classes
        )

        # --- REPLACEMENT ---
        #self.attention_head = TransformerAttention(CHANNELS[name+"-penultimate"], shuffle_times + original_times, last_dim=CHANNELS[name+"-penultimate"]) # <-- output feature dim

        # add a robust MLP classifier after attention
        #self.fc = MLP(
            #input_dim=CHANNELS[name+"-penultimate"],
            #hidden_dims=[512, 256],
            #output_dim=num_classes
        #)

        # freeze CLIP backbone
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def register_hook(self):
        
        def hook(module, input, output):
            self.features = torch.clone(output)
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return 
    
    # -------------------------------
    # Patch Shuffle Operation (PS)
    # -------------------------------
    '''
    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        # Unfold the input tensor to extract non-overlapping patches
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        # Reshape the patches to (B, C, patch_H, patch_W, num_patches)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        # Fold the shuffled patches back into images
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images
    '''
    # -------------------------------
    # Patch Shuffle Operation (PS) + random 90°, 180°, 270° rotations per patch
    # -------------------------------
    def shuffle_patches(self, x, patch_size):
        """
        Randomly shuffle patches of the image and randomly rotate each patch by 90, 180, or 270 degrees.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            patch_size (int): Size of the square patch

        Returns:
            Tensor: Image tensor with shuffled and rotated patches, same shape as input
        """
        B, C, H, W = x.size()

        # === Step 1: Unfold into patches ===
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)  
        # patches shape: (B, C * patch_size * patch_size, num_patches)

        num_patches = patches.size(-1)

        # === Step 2: Reshape patches to 4D (B, num_patches, C, patch_size, patch_size) ===
        patches = patches.transpose(1, 2)  # (B, num_patches, C*ps*ps)
        patches = patches.reshape(B, num_patches, C, patch_size, patch_size)

        # === Step 3: Random shuffle of patch indices ===
        shuffled_indices = torch.randperm(num_patches, device=x.device)
        patches = patches[:, shuffled_indices, :, :, :]

        # === Step 4: Random rotation of each patch ===
        #for b in range(B):
            #for i in range(num_patches):
                #k = torch.randint(0, 4, (1,)).item()  # k ∈ {0,1,2,3} → rotate by k*90°
                #if k > 0:
                    #patches[b, i] = torch.rot90(patches[b, i], k, dims=(1, 2))
        
        # ✅ Step 4: Random rotation + vertical/horizontal flips
        for b in range(B):
            for i in range(num_patches):
                # Rotation
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    patches[b, i] = torch.rot90(patches[b, i], k, dims=(1, 2))

                # Vertical flip
                if torch.rand(1).item() < 0.5:
                    patches[b, i] = torch.flip(patches[b, i], dims=[1])

                # Horizontal flip
                if torch.rand(1).item() < 0.5:
                    patches[b, i] = torch.flip(patches[b, i], dims=[2])

        # === Step 5: Reshape back to (B, C*ps*ps, num_patches) ===
        patches = patches.reshape(B, num_patches, -1).transpose(1, 2)

        # === Step 6: Fold back into full images ===
        shuffled_images = F.fold(
            patches, 
            output_size=(H, W), 
            kernel_size=patch_size, 
            stride=patch_size
        )

        return shuffled_images


    # -------------------------------
    # Forward Pass for 3 Branch
    # -------------------------------
    
    def forward(self, x, return_feature=False):
        features = []
        with torch.no_grad():
            # Patch-shuffled views
            for _ in range(self.shuffle_times):
                self.model.encode_image(self.shuffle_patches(x, patch_size=self.patch_size))
                features.append(self.features.clone())

            # Original image views
            self.model.encode_image(x)
            for _ in range(self.original_times):
                features.append(self.features.clone())

            # === NEW BRANCH: Noise feature ===
            # Step 2: denoise
            '''
            denoised_x = run_denoiser(x)
            noise_feat = run_extractnoise(x, denoised_x)
            # Step 3: extract noise and project to same feature dim as backbone
            global _NOISE_PROJECTOR
            output_dim = CHANNELS[self.name+"-penultimate"]

            # Initialize projector if not already or output_dim changed
            if _NOISE_PROJECTOR is None or _NOISE_PROJECTOR.fc.out_features != output_dim:
                _NOISE_PROJECTOR = NoiseProjector(output_dim=output_dim).to(x.device)
            else:
                _NOISE_PROJECTOR = _NOISE_PROJECTOR.to(x.device)

            # Compute noise feature
            noise_feat = _NOISE_PROJECTOR(noise_feat)   # (B, output_dim)
            features.append(noise_feat)
            '''
            denoised_x = run_denoiser(x)
            # run_extractnoise handles projection internally (returns (B, output_dim))
            noise_feat = run_extractnoise(x, denoised_x, output_dim=CHANNELS[self.name + "-penultimate"])
            features.append(noise_feat)
        # ✅ Convert each feature to 1D DCT
        dct_features = [apply_dct(f) for f in features]
        # ✅ Convert each feature to 2D DCT form
        #dct_features = [apply_2d_dct(f) for f in features]
        # Self-Attention across [e_o, e_s, noise][original, patch-shuffle, noise] without DCT
        #features = self.attention_head(torch.stack(features, dim=-2))
        # Self-Attention across [e_o, e_s, noise] in DCT space
        features = self.attention_head(torch.stack(dct_features, dim=-2))
        return features
    

