import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Noise Projector: supports 1st, 2nd, and 3rd order pooling
# ============================================================

class NoiseProjector(nn.Module):
    """
    Projects noise residuals into feature embeddings using
    1st-, 2nd-, or 3rd-order feature pooling.

    Args:
        input_channels:  number of input channels (default=3)
        output_dim:      target embedding dimension
        order:           pooling order (1, 2, or 3)
    """
    def __init__(self, input_channels=3, output_dim=1024, order=2):
        super().__init__()
        assert order in [1, 2, 3], f"[NoiseProjector] Unsupported order={order}"
        self.order = order

        # --- CNN backbone to compress spatial info ---
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # --- first-order feature (mean) projection ---
        self.fc_mean = nn.Linear(64, output_dim)

        # --- second-order projection ---
        if self.order >= 2:
            self.fc_cov = nn.Linear(64 * 64, output_dim)

        # --- third-order projection ---
        if self.order == 3:
            # 3rd order covariance (64×64×64 flattened)
            self.fc_third = nn.Linear(64 * 64 * 64, output_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"[NoiseProjector] Expected 4D input (B,C,H,W), got {x.shape}")

        feat = self.conv(x)                # (B,64,1,1)
        feat = feat.view(feat.size(0), -1) # (B,64)

        # 1st order
        mean_feat = self.fc_mean(feat)

        # If first-order only
        if self.order == 1:
            return mean_feat

        # 2nd order covariance
        feat_centered = feat - feat.mean(dim=1, keepdim=True)
        cov = torch.bmm(feat_centered.unsqueeze(2), feat_centered.unsqueeze(1))  # (B,64,64)
        cov_flat = cov.reshape(feat.size(0), -1)
        cov_feat = self.fc_cov(cov_flat)

        if self.order == 2:
            return mean_feat + cov_feat

        # 3rd order tensor (outer product of three centered feature vectors)
        # (B,64,1,1)*(B,1,64,1)*(B,1,1,64)
        outer3 = (feat_centered.unsqueeze(2).unsqueeze(3) *
                  feat_centered.unsqueeze(1).unsqueeze(3) *
                  feat_centered.unsqueeze(1).unsqueeze(2))  # (B,64,64,64)

        third_flat = outer3.reshape(feat.size(0), -1)
        third_feat = self.fc_third(third_flat)

        # Combine first + second + third order
        return mean_feat + cov_feat + third_feat


# ============================================================
# Global reusable instance
# ============================================================

_NOISE_PROJECTOR = None


@torch.no_grad()
def run_extractnoise(original_x, denoised_x, output_dim=1024, order=3):
    """
    Compute |original - denoised| and project it into an embedding space
    that includes higher-order pooling (up to 3rd order).

    Args:
        original_x:  torch.Tensor (B,C,H,W)
        denoised_x:  torch.Tensor (B,C,H,W)
        output_dim:  desired feature embedding dimension
        order:       pooling order: 1, 2, or 3

    Returns:
        torch.Tensor (B, output_dim)
    """
    global _NOISE_PROJECTOR
    device = original_x.device

    # Initialize global projector if necessary
    if (
        _NOISE_PROJECTOR is None or
        _NOISE_PROJECTOR.fc_mean.out_features != output_dim or
        _NOISE_PROJECTOR.order != order
    ):
        _NOISE_PROJECTOR = NoiseProjector(output_dim=output_dim, order=order).to(device)
    else:
        _NOISE_PROJECTOR = _NOISE_PROJECTOR.to(device)

    noise = torch.abs(original_x.float() - denoised_x.float())
    return _NOISE_PROJECTOR(noise)
