import torch
import yaml
from util.model_need_tools import set_denoiser

# -----------------------------
# Load denoiser once (global)
# -----------------------------
_DENOISER = None

def load_denoiser(
    pretrained="./ckpt/MMBSN_SIDD_o_a45.pth",
    gpu="cuda",
    config="config/SIDD.yaml"
):
    """
    Initialize the denoiser once and reuse.

    Args:
        pretrained: path to checkpoint
        gpu: either "cuda", "cpu", or GPU index string like "0"
        config: path to YAML config
    Returns:
        denoise_fn: callable denoiser function
    """
    global _DENOISER
    if _DENOISER is None:
        # --- Load YAML ---
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)

        # --- Handle GPU specification ---
        if gpu == "cpu":
            cfg['gpu'] = "cpu"
        else:
            if gpu.isdigit():
                cfg['gpu'] = f"cuda:{gpu}"
            else:
                cfg['gpu'] = gpu

        # --- Ensure pretrained path is in cfg ---
        cfg['pretrained'] = pretrained

        # --- Optional: check if model/kwargs exist ---
        if 'model' not in cfg or 'kwargs' not in cfg['model']:
            raise KeyError("SIDD.yaml must define 'model.kwargs' for BSN initialization!")

        # --- Initialize denoiser ---
        # set_denoiser returns a callable denoise function
        _DENOISER = set_denoiser(checkpoint_path=pretrained, cfg=cfg)
        # ✅ remove .eval() since _DENOISER is a function

    return _DENOISER


@torch.no_grad()
def run_denoiser(x_tensor):
    """
    Run denoiser on a batch of images.

    Args:
        x_tensor: torch.Tensor (B,C,H,W), in [0,1] or [0,255]

    Returns:
        denoised: torch.Tensor (B,C,H,W), same device as input
    """
    device = x_tensor.device
    denoiser_fn = load_denoiser()

    # --- Scale if input in [0,1] ---
    if x_tensor.max() <= 1.0:
        x_tensor = x_tensor * 255.0

    x_tensor = x_tensor.float().to(device)

    # --- Run denoiser ---
    denoised = denoiser_fn(x_tensor)
    return denoised
