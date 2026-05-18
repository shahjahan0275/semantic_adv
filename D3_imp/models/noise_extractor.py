import torch
import torch.nn as nn
'''
# -----------------------------
# Noise Projector (expects (B,C,H,W))
# -----------------------------
class NoiseProjector(nn.Module):
    def __init__(self, input_channels=3, output_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"[NoiseProjector] Expected 4D input (B,C,H,W), got {x.shape}")

        feat = self.conv(x)              # -> (B,64,1,1)
        feat = feat.view(feat.size(0), -1)  # -> (B,64)
        return self.fc(feat)             # -> (B, output_dim)


# Initialize projector globally (will be reused)
_NOISE_PROJECTOR = None

@torch.no_grad()
def run_extractnoise(original_x, denoised_x, output_dim=1024):
    """
    Compute noise = abs(original - denoised) and project into feature space.
    Args:
        original_x: torch.Tensor (B,C,H,W) in [0,255] or normalized
        denoised_x: torch.Tensor (B,C,H,W), same shape
        output_dim: int, feature dimension to match CLIP backbone
    Returns:
        torch.Tensor of shape (B, output_dim)
    """
    global _NOISE_PROJECTOR
    device = original_x.device

    # Initialize or resize projector
    if _NOISE_PROJECTOR is None or _NOISE_PROJECTOR.fc.out_features != output_dim:
        _NOISE_PROJECTOR = NoiseProjector(output_dim=output_dim).to(device)
    else:
        _NOISE_PROJECTOR = _NOISE_PROJECTOR.to(device)

    # Compute raw noise image
    noise = torch.abs(original_x.float() - denoised_x.float())  # (B,C,H,W)

    # Project to feature space
    return _NOISE_PROJECTOR(noise)
'''

# For second order statistics

import torch
import torch.nn as nn

# -----------------------------
# Noise Projector (supports 1st and 2nd order)
# -----------------------------
class NoiseProjector(nn.Module):
    def __init__(self, input_channels=3, output_dim=1024, use_second_order=True):
        super().__init__()
        self.use_second_order = use_second_order
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # first-order feature projection
        self.fc_mean = nn.Linear(64, output_dim)

        if use_second_order:
            # Project second-order (covariance-like) flattened features
            self.fc_cov = nn.Linear(64 * 64, output_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"[NoiseProjector] Expected 4D input (B,C,H,W), got {x.shape}")

        feat = self.conv(x)                # -> (B,64,1,1)
        feat = feat.view(feat.size(0), -1) # -> (B,64)
        mean_feat = self.fc_mean(feat)     # -> (B, output_dim)

        if not self.use_second_order:
            return mean_feat

        # ---- Second-order covariance feature ----
        feat_centered = feat - feat.mean(dim=1, keepdim=True)  # (B,64)
        cov = torch.bmm(feat_centered.unsqueeze(2), feat_centered.unsqueeze(1))  # (B,64,64)
        cov_flat = cov.view(cov.size(0), -1)  # flatten to (B, 4096)
        cov_feat = self.fc_cov(cov_flat)      # project to (B, output_dim)

        # Combine first + second order
        return mean_feat + cov_feat


# -----------------------------
# Global instance for reuse
# -----------------------------
_NOISE_PROJECTOR = None


@torch.no_grad()
def run_extractnoise(original_x, denoised_x, output_dim=1024, use_second_order=True):
    """
    Compute noise = abs(original - denoised) and project into feature space.
    Args:
        original_x: torch.Tensor (B,C,H,W)
        denoised_x: torch.Tensor (B,C,H,W)
        output_dim: int, feature dimension
        use_second_order: bool, include covariance features
    Returns:
        torch.Tensor (B, output_dim)
    """
    global _NOISE_PROJECTOR
    device = original_x.device

    if _NOISE_PROJECTOR is None or \
       _NOISE_PROJECTOR.fc_mean.out_features != output_dim or \
       _NOISE_PROJECTOR.use_second_order != use_second_order:
        _NOISE_PROJECTOR = NoiseProjector(output_dim=output_dim, use_second_order=use_second_order).to(device)
    else:
        _NOISE_PROJECTOR = _NOISE_PROJECTOR.to(device)

    noise = torch.abs(original_x.float() - denoised_x.float())
    return _NOISE_PROJECTOR(noise)
