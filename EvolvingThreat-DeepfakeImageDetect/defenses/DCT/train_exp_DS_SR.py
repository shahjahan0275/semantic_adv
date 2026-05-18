# Data from only one training and one validation generator

# Training code without patch shuffling (full 1024×1024 images)

import os
import torch
import argparse
import random 
import numpy as np
import torch_dct as dct
from pathlib import Path
from PIL import Image 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Add near the top of your file with other imports:
import torch.nn.functional as F
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False
    # The code below will raise a clear error if pywt is required but missing.
DEVICE = "cuda:0"
# -----------------------------------------------------------------------------
# Combined Spatio-Frequency feature extractor
# -----------------------------------------------------------------------------
def compute_spatio_frequency_features(
    x,
    wavelet='db2',
    wavelet_levels=3,
    downsample_size=(64, 64),
    eps=1e-12
):
    """
    Compute combined spatio-frequency features for a batch of images.

    Inputs:
      x: Tensor shape (N, C, H, W) in range [-1, 1]
      wavelet: wavelet name for wavedec2 (pywt)
      wavelet_levels: number of decomposition levels for 2D wavelet
      downsample_size: (H_out, W_out) to spatially reduce the maps
      eps: numerical eps to stabilize logs

    Returns:
      features: Tensor shape (N, C_combined, H_out, W_out) where C_combined includes
                [power, autocorr, wavelet-energy-maps_per_channel]
                The caller often will then squeeze channel=1 and flatten per-sample.
    """
    if not _HAS_PYWT:
        raise RuntimeError("pywt (PyWavelets) is required for wavelet decomposition. "
                           "Install it with `pip install pywavelets` on your cluster.")

    N, C, H, W = x.shape
    device = x.device

    # ---------- 1) Power Spectrum (per channel) ----------
    # Compute 2D FFT along spatial dims
    fft2d = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    power = fft2d.abs() ** 2  # shape (N, C, H, W)
    power_log = torch.log1p(power)  # log(1 + power)

    # ---------- 2) Spatial Autocorrelation via inverse FFT (Wiener-Khinchin) ----------
    # Autocorr = IFFT( power ) ; keep real part
    autocorr = torch.fft.ifftn(power, dim=(-2, -1)).real  # shape (N, C, H, W)
    # Move zero-lag to center for visualization & stability (optional)
    autocorr_shifted = torch.fft.fftshift(autocorr, dim=(-2, -1))
    autocorr_log = torch.log1p(torch.abs(autocorr_shifted) + eps)

    # ---------- 3) Wavelet decomposition energies (per channel) ----------
    # We'll compute level-wise detail energies and put them into maps.
    # For simplicity, create a "wavelet energy image" per channel by upsampling each subband energy
    # to (H, W) and summing/stacking them.
    wavelet_maps = []
    for n in range(N):
        # per-sample
        sample_maps = []
        for ch in range(C):
            data = x[n, ch].cpu().numpy()  # pywt expects numpy 2D array
            coeffs = pywt.wavedec2(data, wavelet, level=wavelet_levels)
            # coeffs: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, ...), ...]
            # We'll compute L2-energy of each subband and map it back to spatial grid by
            # placing a constant map of the subband energy scaled to subband shape
            # then upsample to HxW and sum to produce a single energy image per channel.
            ch_maps = []
            # approx coeff
            cA = coeffs[0]
            energy_cA = (cA ** 2).sum()
            # make constant map same spatial shape as cA then upsample
            map_cA = torch.full((1, cA.shape[0], cA.shape[1]), float(energy_cA), device='cpu')
            up_cA = F.interpolate(map_cA.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
            ch_maps.append(up_cA[0])  # (1, H, W) → (1, H, W) preserved


            # detail coeffs
            for lvl_details in coeffs[1:]:
                cH, cV, cD = lvl_details
                for csub in (cH, cV, cD):
                    energy = (csub ** 2).sum()
                    map_sub = torch.full((1, csub.shape[0], csub.shape[1]), float(energy), device='cpu')
                    up = F.interpolate(map_sub.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
                    ch_maps.append(up[0])


            # stack all upsampled subband-energy maps for this channel and sum or keep them stacked:
            # Here we'll stack so we preserve different subband energies as separate channels.
            
            ch_maps_tensor = torch.stack(ch_maps, dim=0).squeeze(1)  # (num_subbands, H, W)

            # Option A: sum across subbands -> single map per channel
            ch_sum = ch_maps_tensor.sum(dim=0)  # (H, W)
            # Option B (commented): keep all subband maps individually (increase channels)
            # ch_stacked = ch_maps_tensor

            sample_maps.append(ch_sum.to(device))   # shape (H, W) on device

        # stack channels for this sample: shape (C, H, W)
        sample_wave_maps = torch.stack(sample_maps, dim=0)
        wavelet_maps.append(sample_wave_maps)

    wavelet_maps = torch.stack(wavelet_maps, dim=0)  # (N, C, H, W)
    print("DEBUG shapes → power:", power_log.shape, "autocorr:", autocorr_log.shape, "wavelet:", wavelet_maps.shape)


    # ---------- 4) Concatenate features along channel axis ----------
    # We'll concatenate [power_log, autocorr_log, wavelet_maps] -> channels = 3*C
    combined = torch.cat([power_log, autocorr_log, wavelet_maps], dim=1)  # (N, 3*C, H, W)

    # ---------- 5) Spatial downsample to reduce dimensionality ----------
    if downsample_size is not None:
        H_out, W_out = downsample_size
        combined = F.interpolate(combined, size=(H_out, W_out), mode='bilinear', align_corners=False)

    # ---------- 6) Optional final log / normalization (caller will standardize) ----------
    # Already used log1p for power & autocorr; wavelet_maps are energies (positive); we can log them too:
    combined = torch.log1p(combined.abs())

    return combined  # shape (N, C_combined, H_out, W_out)

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def dct_3d(x, norm='ortho'):
    """
    Applies 3D DCT by applying 1D DCT along each axis (C, H, W).
    Expects input of shape (N, C, H, W).
    """
    # Apply along width (dim=-1)
    x = dct.dct(x, norm=norm)
    # Apply along height (dim=-2)
    x = dct.dct(x.transpose(-1, -2), norm=norm).transpose(-1, -2)
    # Apply along channel (dim=1)
    x = dct.dct(x.transpose(1, -1), norm=norm).transpose(1, -1)
    return x

def power_spectrum_3d(x):
    """
    Computes 3D Power Spectrum (second-order frequency-domain features).
    Expects input of shape (N, C, H, W).
    Returns log-scaled power spectrum.
    """
    # Apply 2D FFT over spatial dimensions (H, W)
    fft2d = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')

    # Power spectrum = magnitude squared
    power = fft2d.abs() ** 2

    # Log-compression for dynamic range stabilization
    power_log = torch.log1p(power)

    return power_log

def visualize_spatio_frequency_components(img_tensor, downsample_size=(256,256), wavelet='db2', wavelet_levels=3, title_prefix="Sample"):
    """
    Shows power, autocorr, and wavelet-energy maps side-by-side for one image.
    img_tensor: (C, H, W) or (1, C, H, W)
    """
    if img_tensor.ndim == 4:
        img = img_tensor[0]
    else:
        img = img_tensor

    # ensure batch dim for compute function
    x = img.unsqueeze(0)  # (1, C, H, W)

    # compute combined features at a reasonably large downsample for visualization
    comb = compute_spatio_frequency_features(x, wavelet=wavelet, wavelet_levels=wavelet_levels, downsample_size=downsample_size)
    # comb shape (1, 3*C, H_out, W_out)
    comb = comb[0].cpu().numpy()  # (3*C, H, W)

    C = img.shape[0]
    # For grayscale: indices 0=power,1=autocorr,2=wavelet. For RGB: blocks of 3.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Merge across channels for visualization (sum)
    # Power:
    power_map = comb[0:C].sum(axis=0)
    autocorr_map = comb[C:2*C].sum(axis=0)
    wavelet_map = comb[2*C:3*C].sum(axis=0)

    axs[0].imshow(power_map, cmap='inferno')
    axs[0].set_title(f"{title_prefix} - Power Spectrum (log)")
    axs[0].axis('off')

    axs[1].imshow(autocorr_map, cmap='inferno')
    axs[1].set_title(f"{title_prefix} - Autocorrelation (log)")
    axs[1].axis('off')

    axs[2].imshow(wavelet_map, cmap='inferno')
    axs[2].set_title(f"{title_prefix} - Wavelet Energy (log)")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def save_spatio_freq_visual(x, save_path, title="Spatio-Frequency Map"):
    """
    Saves a visualization of the spatio-frequency representation (not interactive).
    Expects x as 2D or 3D tensor/array. If 3D, it collapses across channels.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    if x.ndim == 3:
        # collapse channel dimension (C, H, W) → (H, W)
        x = x.mean(axis=0)

    plt.figure(figsize=(6,6))
    plt.imshow(x, cmap='inferno')
    plt.title(title)
    plt.axis('off')
    plt.colorbar(label="Energy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def array_from_imgdir(imgdir, grayscale=True, max_samples=None, seed=42):
    """
    Loads images from a directory, center-crops to 1024×1024,
    converts to tensor (no patch shuffling).
    """
    imgnames = os.listdir(imgdir)
    random.seed(seed)
    if max_samples is not None:
        imgnames = random.sample(imgnames, min(max_samples, len(imgnames)))

    crop_tf = transforms.CenterCrop((1024, 1024))

    def loader(path):
        img = Image.open(path)
        img = img.convert("L") if grayscale else img.convert("RGB")
        img = crop_tf(img)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = (img_tensor * 2.0) - 1.0  # scale to [-1, 1]
        return img_tensor

    paths = [os.path.join(imgdir, img) for img in imgnames]
    array = torch.stack(Parallel(n_jobs=4)(delayed(loader)(p) for p in paths))
    print(f"Loaded {len(array)} images (no patch shuffle) from {imgdir}")
    return array


# define a logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(LogisticRegression, self).__init__() 
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x): 
        out1 = self.relu(self.linear1(x))
        out2 = self.relu(self.linear2(out1))
        out3 = self.linear3(out2)
        return out3


class MyDataset(Dataset):
    def __init__(self, x_data, labels):
        self.x_data = x_data
        self.labels = labels
  
    def __len__(self):
        return len(self.x_data)
  
    def __getitem__(self, idx): 
        x = self.x_data[idx]
        #y = torch.tensor(self.labels[idx])
        y = self.labels[idx].clone().detach()
        return x, y


def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_acc += (preds == labels).float().mean().item()
    
    return train_loss / len(train_loader), train_acc / len(train_loader)


def valid_epoch(model, criterion, valid_loader):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).long()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            valid_acc += (preds == labels).float().mean().item()
    
    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)



def main(args):
    # ===================== 🔹 STEP 1: Load TRAIN and VALIDATION data (no patch shuffle) =====================
    real_train = array_from_imgdir(
        args.image_root / "train" / "real",
        max_samples=args.num_real_train
    )
    fake_train = array_from_imgdir(
        args.image_root / "train" / "fake",
        max_samples=args.num_fake_train
    )
    print(f"Selected {len(real_train)} real and {len(fake_train)} fake samples for TRAINING.")
    
    # After loading real_train and fake_train:
    print("Visualizing spatio-frequency components for a real sample...")
    visualize_spatio_frequency_components(real_train[0], downsample_size=(256,256), wavelet='db2', wavelet_levels=3, title_prefix="Real")

    print("Visualizing spatio-frequency components for a fake sample...")
    visualize_spatio_frequency_components(fake_train[0], downsample_size=(256,256), wavelet='db2', wavelet_levels=3, title_prefix="Fake")



    real_val = array_from_imgdir(
        args.image_root / "val" / "real",
        max_samples=args.num_real_val
    )
    fake_val = array_from_imgdir(
        args.image_root / "val" / "fake",
        max_samples=args.num_fake_val
    )
    print(f"Selected {len(real_val)} real and {len(fake_val)} fake samples for VALIDATION.")

    # ===================== 🔹 Combine and process data =====================
    x_train = torch.cat([real_train, fake_train], dim=0)
    y_train = torch.tensor([0.0] * len(real_train) + [1.0] * len(fake_train))
    del real_train, fake_train

    x_val = torch.cat([real_val, fake_val], dim=0)
    y_val = torch.tensor([0.0] * len(real_val) + [1.0] * len(fake_val))
    del real_val, fake_val

    # ===================== 🔹 STEP 2: Spatio-frequency computation and normalization =====================
    print('feature calculation...')

    # Use the combined extractor. You can tune wavelet_levels and downsample_size.
    print("Computing training features in smaller batches to save memory...")
    chunk_size = 4  # or 2 if memory is still low
    feats = []
    for i in range(0, len(x_train), chunk_size):
        x_chunk = x_train[i:i+chunk_size].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            f = compute_spatio_frequency_features(
                x_chunk,
                wavelet='db2',
                wavelet_levels=3,
                downsample_size=(64, 64)
            )
        feats.append(f.cpu())
        del f, x_chunk
        torch.cuda.empty_cache()

    combined_train = torch.cat(feats, dim=0)
    del feats
    torch.cuda.empty_cache()
    print(f"Training features computed: {combined_train.shape}")

    print("Computing validation features in smaller batches to save memory...")
    chunk_size = 4  # or 2 if memory is still low
    feats = []
    
    for i in range(0, len(x_val), chunk_size):
        x_chunk = x_val[i:i + chunk_size].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            f = compute_spatio_frequency_features(
                x_chunk,
                wavelet='db2',
                wavelet_levels=3,
                downsample_size=(64, 64)
            )
        feats.append(f.cpu())
        del f, x_chunk
        torch.cuda.empty_cache()

    combined_val = torch.cat(feats, dim=0)
    del feats
    torch.cuda.empty_cache()
    print(f"Validation features computed: {combined_val.shape}")
    # We'll convert to single-channel-like features by collapsing channel dim:
    # If you want to keep spatial maps, remove the squeeze() and flatten accordingly.
    # Here we assume grayscale (C=1) or you accept multi-channel flattened vector.
    N_train = combined_train.shape[0]
    N_val = combined_val.shape[0]

    # Flatten per sample for logistic model
    x_train_tf = combined_train.reshape(N_train, -1)
    x_val_tf = combined_val.reshape(N_val, -1)
    print('reshaped...')
   
    means = x_train_tf.mean(0, keepdim=True)
    stds = x_train_tf.std(0, unbiased=False, keepdim=True)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(means, os.path.join(args.save_dir, 'means.pt'))
    torch.save(stds, os.path.join(args.save_dir, 'stds.pt'))

    x_train_tf = (x_train_tf - means) / stds
    x_val_tf = (x_val_tf - means) / stds
    print("rescaled...")

    # ===================== 🔹 STEP 3: Model Training =====================
    train_dataset = MyDataset(x_train_tf, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = MyDataset(x_val_tf, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    print('training model...')

    # Infer the correct flattened input size dynamically
    input_size = x_train_tf.shape[1]  # number of flattened features per image
    print(f"Detected input feature size: {input_size}")

    model = LogisticRegression(input_size, num_classes=2)
    model.to(DEVICE)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        valid_loss, valid_acc = valid_epoch(model, criterion, val_loader)

        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}')
        
        # ✅ Visualization checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Compute spatio-frequency map for a validation sample
                spf_map = compute_spatio_frequency_features(
                    x_val[0:1],
                    wavelet='db2',
                    wavelet_levels=3,
                    downsample_size=(64, 64)
                )
                spf_np = spf_map.squeeze().cpu().numpy()
                save_path = os.path.join(args.save_dir, f"spatio_freq_epoch{epoch+1}.png")
                save_spatio_freq_visual(spf_np, save_path)
                print(f"🖼️ Saved spatio-frequency visualization: {save_path}")

        # ✅ Save best-performing model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_Mj.pth'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=Path,
        help="Root of image directory containing 'train', 'val', and test.",
    )
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="checkpoints_rgb")
    parser.add_argument("--num_real_train", type=int, default=400)
    parser.add_argument("--num_fake_train", type=int, default=400)
    parser.add_argument("--num_real_val", type=int, default=50)
    parser.add_argument("--num_fake_val", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()
    main(parse_args())

