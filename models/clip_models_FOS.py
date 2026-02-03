from .clip import clip 
from PIL import Image
import torch.nn as nn
import math

import torch
import torch.nn.functional as F
import torch.fft as fft
from models.transformer_attention import TransformerAttention
import torchvision.transforms as transforms
from .clip.model import VisionTransformer
from .mlp import MLP

from .denoiser import run_denoiser
from .noise_extractor_FOS import run_extractnoise, NoiseProjector

_NOISE_PROJECTOR = None

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "ViT-L/14-penultimate": 1024
}

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

# =========================================================
# Base CLIP models (unchanged)
# =========================================================

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

class CLIPModelPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModelPenultimateLayer, self).__init__()
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.register_hook()
        self.fc = nn.Linear(CHANNELS[name+"-penultimate"], num_classes)

    def register_hook(self):
        def hook(module, input, output):
            self.features = torch.clone(output)
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)

    def forward(self, x):
        self.model.encode_image(x)
        return self.fc(self.features)


# =========================================================
# Higher-Order Pooling Modules (2nd, 3rd, 4th)
# =========================================================

class SecondOrderPooling(nn.Module):
    def __init__(self, input_dim, output_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is not None:
            self.proj = nn.Linear(input_dim * input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x):
        B, N, D = x.shape
        mean = x.mean(dim=1, keepdim=True)
        xc = x - mean
        cov = torch.bmm(xc.transpose(1, 2), xc) / max(1, (N - 1))
        cov_flat = cov.reshape(B, -1)
        if self.proj is not None:
            cov_flat = self.proj(cov_flat)
        return cov_flat

class ThirdOrderPooling(nn.Module):
    def __init__(self, input_dim, output_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is not None:
            self.proj = nn.Linear(input_dim ** 3, output_dim)
        else:
            self.proj = None

    def forward(self, x):
        B, N, D = x.shape
        mean = x.mean(dim=1, keepdim=True)
        xc = x - mean
        outer = (xc.unsqueeze(3).unsqueeze(4) *
                 xc.unsqueeze(2).unsqueeze(4) *
                 xc.unsqueeze(2).unsqueeze(3))
        third = outer.mean(dim=1)
        third_flat = third.reshape(B, -1)
        if self.proj is not None:
            third_flat = self.proj(third_flat)
        return third_flat

class FourthOrderPooling(nn.Module):
    """
    Memory-efficient approximation of 4th-order statistics.

    Strategy:
      - Compute per-dimension 4th central moment (kurtosis-like): mean((x - mean)^4, dim=1) -> (B, D)
      - Compute covariance of squared centered features:
            sq = (x - mean)**2  -> (B, N, D)
            cov_sq = (sq^T @ sq) / (N-1)  -> (B, D, D)
        Flatten cov_sq (size D*D) and project it down.
      - Concatenate [kurtosis, projected cov_sq] and linearly project to `output_dim`.

    Complexity:
      - Memory/time: O(B * D^2) for cov_sq (with D = input_dim). This is feasible for input_dim ~= 64..256.
      - Avoids impossible O(D^4) allocations.
    """
    def __init__(self, input_dim, output_dim=None, eps=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps

        if self.output_dim is not None:
            # We'll compute kurtosis (size=input_dim) + a projection of cov_sq
            # Decide how many dims of cov_sq projection we want:
            cov_proj_dim = max(1, self.output_dim - self.input_dim)  # ensure at least 1
            self.proj_cov = nn.Linear(input_dim * input_dim, cov_proj_dim)
            # Final linear layer to produce exactly output_dim
            self.final = nn.Linear(self.input_dim + cov_proj_dim, self.output_dim)
        else:
            self.proj_cov = None
            self.final = None

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        mean = x.mean(dim=1, keepdim=True)          # (B,1,D)
        xc = x - mean                               # (B,N,D)

        # 1) per-dimension 4th central moment (kurtosis-like)
        kurtosis = (xc.pow(4).mean(dim=1))          # (B, D)

        if self.output_dim is None:
            # Return raw kurtosis flattened if no projection requested
            return kurtosis

        # 2) covariance of squared centered features: captures many 4th-order cross-terms
        sq = xc.pow(2)                              # (B, N, D)
        # compute cov of squared features: (B, D, D)
        # center sq along N to reduce bias
        sq_mean = sq.mean(dim=1, keepdim=True)      # (B,1,D)
        sqc = sq - sq_mean                          # (B,N,D)
        cov_sq = torch.bmm(sqc.transpose(1, 2), sqc) / max(1, (N - 1))  # (B, D, D)
        cov_sq_flat = cov_sq.reshape(B, -1)         # (B, D*D)

        # project cov_sq_flat -> cov_proj (B, cov_proj_dim)
        cov_proj = self.proj_cov(cov_sq_flat)       # (B, cov_proj_dim)

        # concat kurtosis and projected covariances and final projection -> output_dim
        combined = torch.cat([kurtosis, cov_proj], dim=1)  # (B, D + cov_proj_dim)
        out = self.final(combined)                   # (B, output_dim)
        return out

class MomentPooling(nn.Module):
    """
    Computes mean, variance, skewness, kurtosis over tokens.
    Input:  (B, N, D)
    Output: (B, 4D)
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, N, D)
        mean = x.mean(dim=1)                          # (B, D)

        xc = x - mean.unsqueeze(1)
        var = (xc ** 2).mean(dim=1)                   # (B, D)
        std = torch.sqrt(var + self.eps)

        skew = (xc ** 3).mean(dim=1) / (std ** 3 + self.eps)
        kurt = (xc ** 4).mean(dim=1) / (std ** 4 + self.eps)

        # (B, 4D)
        return torch.cat([mean, var, skew, kurt], dim=1)



class DCTMomentPooling(nn.Module):
    """
    True DCT-II based moment pooling.
    Computes mean, variance, skewness, kurtosis
    over DCT coefficients along token dimension.

    Input:  (B, N, D)
    Output: (B, 4D)
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("dct_mat", None)

    def _build_dct(self, N, device):
        """
        Create DCT-II transform matrix of size (N, N)
        """
        k = torch.arange(N, device=device).float().unsqueeze(1)
        n = torch.arange(N, device=device).float().unsqueeze(0)

        dct = torch.cos(math.pi / N * (n + 0.5) * k)

        dct[0] *= math.sqrt(1.0 / N)
        dct[1:] *= math.sqrt(2.0 / N)

        return dct

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        if self.dct_mat is None or self.dct_mat.shape[0] != N:
            self.dct_mat = self._build_dct(N, x.device)

        # Apply DCT-II along token dimension
        # (B, N, D) -> (B, N, D)
        x_dct = torch.einsum("kn,bnd->bkd", self.dct_mat, x)

        # Moments over DCT coefficients
        mean = x_dct.mean(dim=1)                # (B, D)
        xc = x_dct - mean.unsqueeze(1)

        var = (xc ** 2).mean(dim=1)
        std = torch.sqrt(var + self.eps)

        skew = (xc ** 3).mean(dim=1) / (std ** 3 + self.eps)
        kurt = (xc ** 4).mean(dim=1) / (std ** 4 + self.eps)

        return torch.cat([mean, var, skew, kurt], dim=1)


# =========================================================
# Main Model: CLIP + Shuffle + Attention + 4th Order Pooling
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

        self.model, self.preprocess = clip.load(name, device="cpu")
        self.register_hook()

        num_branches = self.shuffle_times + self.original_times + 1
        D = CHANNELS[self.name + "-penultimate"]

        self.attention_head = TransformerAttention(
            input_dim=D,
            num_branches=num_branches,
            last_dim=D
        )

        # --- Use reduction before 4th-order pooling ---
        '''
        self.reduction = nn.Linear(D, D // 8, bias=False)
        self.fourth_order_pool = FourthOrderPooling(
            input_dim=D // 8,
            output_dim=D
        )

        self.fc = nn.Linear(2 * D, self.num_classes)
        '''
        # Reduce token dimension before statistics
        self.reduction = nn.Linear(D, D // 4, bias=False)

        # Moment pooling (mean, var, skew, kurt)
        #self.moment_pool = MomentPooling()

        #----- DCT-based moment pooling ----------#
        self.dct_moment_pool = DCTMomentPooling()
        #-----------------------------------------#


        # Moment output size = 4 * (D//4) = D
        moment_dim = D

        # Attention still outputs D
        self.fc = nn.Linear(D + moment_dim, self.num_classes)

        for p in self.model.parameters():
            p.requires_grad = False

    def register_hook(self):
        def hook(module, input, output):
            self.features = torch.clone(output)
        for child_name, module in self.model.visual.named_children():
            if child_name == "ln_post":
                module.register_forward_hook(hook)

    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        num_patches = patches.size(-1)
        patches = patches.transpose(1, 2)
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
        features = []
        with torch.no_grad():
            for _ in range(self.shuffle_times):
                self.model.encode_image(self.shuffle_patches(x, self.patch_size))
                features.append(self.features.clone())

            self.model.encode_image(x)
            for _ in range(self.original_times):
                features.append(self.features.clone())

            denoised_x = run_denoiser(x)
            clip_dim = CHANNELS[self.name + "-penultimate"]
            noise_feat = run_extractnoise(x, denoised_x, output_dim=clip_dim, order=4)
            features.append(noise_feat)

        features_stack = torch.stack(features, dim=1)

        #reduced = self.reduction(features_stack)
        #fourth_order_features = self.fourth_order_pool(reduced)
        # (B, num_branches, D)
        reduced = self.reduction(features_stack)  # (B, N, D//4)

        # Compute all moments
        #moment_features = self.moment_pool(reduced)  # (B, D)
        
        #--------- DCT-based 4th-order moments-----------#
        moment_features = self.dct_moment_pool(reduced)  # (B, D)

        moment_features = F.normalize(moment_features, dim=1)


        attn_out = self.attention_head(features_stack)
        if attn_out.dim() == 3:
            attn_out = attn_out.mean(dim=1)

        attn_out = F.normalize(attn_out, dim=1)
        #fourth_order_features = F.normalize(fourth_order_features, dim=1)

        #fused = torch.cat([attn_out, fourth_order_features], dim=1)
        fused = torch.cat([attn_out, moment_features], dim=1)
        logits = self.fc(fused)

        if return_feature:
            return fused
        return logits
