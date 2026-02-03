from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.transformer_attention import TransformerAttention
import torchvision.transforms as transforms
from .clip.model import VisionTransformer
from .mlp import MLP

from .denoiser import run_denoiser       # wrapper for test.py logic (Step 2)    3 branch
from .noise_extractor_TOS import run_extractnoise, NoiseProjector  # wrapper for extractnoise.py logic (Step 3)   3 branch

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

# =========================================================
# CLIP:ViT-L/14 Feature Extractor (Visual Backbone)
# =========================================================

class CLIPModel(nn.Module):
    """UFD"""
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )

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



class SecondOrderPooling(nn.Module):
    """
    Compute second-order (covariance-based) feature statistics.
    Input: (B, N, D) — N branches or spatial tokens, D feature dimension
    Output: (B, proj_dim) where proj_dim = output_dim if provided,
            otherwise returns flattened covariance (B, D*D).
    """
    def __init__(self, input_dim, output_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is not None:
            # project flattened covariance (D*D) -> output_dim
            self.proj = nn.Linear(input_dim * input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        if N <= 1:
            # Avoid divide-by-zero: if only one branch, return zeros or pooled features
            # We'll return flattened outer product of (x - mean) which will be zeros.
            mean = x.mean(dim=1, keepdim=True)  # (B,1,D)
            xc = x - mean
            cov = torch.bmm(xc.transpose(1, 2), xc) / max(1, (N - 1))
        else:
            mean = x.mean(dim=1, keepdim=True)   # (B,1,D)
            xc = x - mean                         # (B,N,D)
            cov = torch.bmm(xc.transpose(1, 2), xc) / (N - 1)  # (B, D, D)

        cov_flat = cov.reshape(B, -1)  # (B, D*D)
        if self.proj is not None:
            cov_flat = self.proj(cov_flat)  # (B, output_dim)
        return cov_flat  # (B, output_dim or D*D)

class ThirdOrderPooling(nn.Module):
    """
    Compute third-order (central-moment) feature statistics.
    Input: (B, N, D) — N branches or spatial tokens, D feature dimension
    Output: (B, output_dim) where output_dim = provided, else D^3 (flattened tensor)
    """
    def __init__(self, input_dim, output_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is not None:
            # project flattened 3rd-order tensor (D^3) -> output_dim
            self.proj = nn.Linear(input_dim * input_dim * input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        mean = x.mean(dim=1, keepdim=True)     # (B,1,D)
        xc = x - mean                          # center

        # --- compute third-order tensor ---
        # outer product: (B,N,D,1,1)*(B,N,1,D,1)*(B,N,1,1,D)
        outer = (xc.unsqueeze(3).unsqueeze(4) *
                 xc.unsqueeze(2).unsqueeze(4) *
                 xc.unsqueeze(2).unsqueeze(3))  # (B,N,D,D,D)

        third_order = outer.mean(dim=1)        # average over branches -> (B,D,D,D)
        third_order_flat = third_order.reshape(B, -1)  # (B, D^3)

        if self.proj is not None:
            third_order_flat = self.proj(third_order_flat)
        return third_order_flat


# =========================================================
# Patch Shuffle + Self-Attention Invariance Extraction
# =========================================================

class CLIPModelShuffleAttentionPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1, shuffle_times=1, patch_size=32, original_times=1):
        super(CLIPModelShuffleAttentionPenultimateLayer, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.shuffle_times = shuffle_times
        self.original_times = original_times

        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        if self.patch_size <= 1:
            print(f"[clip_models] Warning: invalid patch_size={self.patch_size}, resetting to 14")
            self.patch_size = 14

        # load CLIP backbone and hook penultimate features
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.register_hook()

        # number of branches (shuffled + original + noise)
        num_branches = self.shuffle_times + self.original_times + 1

        # attention head: keeps existing API (accepts (B, N, D) typically)
        self.attention_head = TransformerAttention(
            input_dim=CHANNELS[self.name + "-penultimate"],
            num_branches=num_branches,
            last_dim=CHANNELS[self.name + "-penultimate"]  # we want an output with same D so we can fuse
        )

        
        # third-order pooling: project D^3 -> D
        D = CHANNELS[self.name + "-penultimate"]
        self.reduction = nn.Linear(D, D // 8, bias=False)
        self.third_order_pool = ThirdOrderPooling(
            input_dim=D // 8,   # compress before forming tensor
            output_dim=D
        )
        
        # final classifier: will take concatenated [attn_out (B,D), second_order (B,D)] => 2D
        self.fc = nn.Linear(2 * D, self.num_classes)

        # freeze CLIP backbone
        for p_name, param in self.model.named_parameters():
            param.requires_grad = False

    def register_hook(self):
        def hook(module, input, output):
            # capture the penultimate features (usually ln_post output)
            self.features = torch.clone(output)
        for child_name, module in self.model.visual.named_children():
            if child_name == "ln_post":
                module.register_forward_hook(hook)
        return

    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        num_patches = patches.size(-1)
        patches = patches.transpose(1, 2)  # (B, num_patches, C*ps*ps)
        patches = patches.reshape(B, num_patches, C, patch_size, patch_size)
        shuffled_indices = torch.randperm(num_patches, device=x.device)
        patches = patches[:, shuffled_indices, :, :, :]
        for b in range(B):
            for i in range(num_patches):
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    patches[b, i] = torch.rot90(patches[b, i], k, dims=(1, 2))
                if torch.rand(1).item() < 0.5:
                    patches[b, i] = torch.flip(patches[b, i], dims=[1])
                if torch.rand(1).item() < 0.5:
                    patches[b, i] = torch.flip(patches[b, i], dims=[2])
        patches = patches.reshape(B, num_patches, -1).transpose(1, 2)
        shuffled_images = F.fold(patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images

    def forward(self, x, return_feature=False):
        """
        1) Build raw branch features list (each element shape (B, D))
        2) Stack -> (B, N, D)
        3) second-order pool on stack -> (B, D)
        4) attention_head on stack -> attn_out (B, D)  (if attention returns (B,N,D) we mean-pool)
        5) fuse = concat(attn_out, second_order) -> (B, 2D) -> fc -> logits
        """
        features = []
        with torch.no_grad():
            # patch-shuffled views
            for _ in range(self.shuffle_times):
                self.model.encode_image(self.shuffle_patches(x, patch_size=self.patch_size))
                features.append(self.features.clone())   # (B, D)

            # original image views
            self.model.encode_image(x)
            for _ in range(self.original_times):
                features.append(self.features.clone())   # (B, D)

            # noise branch : run denoiser + extract noise projected to D
            # -------------------------------
            # Noise branch (with second-order enhancement)
            # -------------------------------
            denoised_x = run_denoiser(x)
            clip_dim = CHANNELS[self.name + "-penultimate"]
            # Compute enhanced noise embedding (includes 2nd-order stats inside run_extractnoise)
            noise_feat = run_extractnoise(
                x, denoised_x,
                output_dim=clip_dim,
                order=3  # ✅ use third-order pooling in the noise projector
            )
            features.append(noise_feat)  # (B, D)


        # Stack into (B, N, D)
        features_stack = torch.stack(features, dim=1)  # branches dim=1

        # 1) third-order features from raw branches
        reduced = self.reduction(features_stack)  # (B, N, D//8)
        third_order_features = self.third_order_pool(reduced)  # (B, D)

        # 2) attention output
        attn_out = self.attention_head(features_stack)  # may return (B, D) OR (B, N, D)
        if attn_out.dim() == 3:
            # pool across branch dimension if returned per-branch
            attn_out = attn_out.mean(dim=1)  # (B, D)
        
        # normalize both before fusion
        attn_out = F.normalize(attn_out, dim=1)
        third_order_features = F.normalize(third_order_features, dim=1)

        # 3) fuse and classify
        #fused = torch.cat([attn_out, second_order_features], dim=1)  # (B, 2D)
        fused = torch.cat([attn_out, third_order_features], dim=1)
        logits = self.fc(fused)



        if return_feature:
            # optionally return fused features (before fc)
            return fused
        return logits

