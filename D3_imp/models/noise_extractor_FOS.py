import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseProjector(nn.Module):
    def __init__(self, input_channels=3, output_dim=1024, order=2):
        super().__init__()
        assert order in [1, 2, 3, 4], f"[NoiseProjector] Unsupported order={order}"
        self.order = order

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_mean = nn.Linear(64, output_dim)
        if self.order >= 2:
            self.fc_cov = nn.Linear(64 * 64, output_dim)
        if self.order >= 3:
            self.fc_third = nn.Linear(64 ** 3, output_dim)
        if self.order == 4:
            self.fc_fourth = nn.Linear(64 ** 4, output_dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"[NoiseProjector] Expected 4D input (B,C,H,W), got {x.shape}")

        feat = self.conv(x).view(x.size(0), -1)
        mean_feat = self.fc_mean(feat)

        if self.order == 1:
            return mean_feat

        feat_centered = feat - feat.mean(dim=1, keepdim=True)

        cov = torch.bmm(feat_centered.unsqueeze(2), feat_centered.unsqueeze(1))
        cov_feat = self.fc_cov(cov.reshape(feat.size(0), -1))

        if self.order == 2:
            return mean_feat + cov_feat

        outer3 = (feat_centered.unsqueeze(2).unsqueeze(3) *
                  feat_centered.unsqueeze(1).unsqueeze(3) *
                  feat_centered.unsqueeze(1).unsqueeze(2))
        third_feat = self.fc_third(outer3.reshape(feat.size(0), -1))

        if self.order == 3:
            return mean_feat + cov_feat + third_feat

        # Fourth-order statistics
        outer4 = (feat_centered.unsqueeze(2).unsqueeze(3).unsqueeze(4) *
                  feat_centered.unsqueeze(1).unsqueeze(3).unsqueeze(4) *
                  feat_centered.unsqueeze(1).unsqueeze(2).unsqueeze(4) *
                  feat_centered.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        fourth_feat = self.fc_fourth(outer4.reshape(feat.size(0), -1))

        return mean_feat + cov_feat + third_feat + fourth_feat


_NOISE_PROJECTOR = None

@torch.no_grad()
def run_extractnoise(original_x, denoised_x, output_dim=1024, order=4):
    global _NOISE_PROJECTOR
    device = original_x.device

    if (_NOISE_PROJECTOR is None or
        _NOISE_PROJECTOR.fc_mean.out_features != output_dim or
        _NOISE_PROJECTOR.order != order):
        _NOISE_PROJECTOR = NoiseProjector(output_dim=output_dim, order=order).to(device)
    else:
        _NOISE_PROJECTOR = _NOISE_PROJECTOR.to(device)

    noise = torch.abs(original_x.float() - denoised_x.float())
    return _NOISE_PROJECTOR(noise)
