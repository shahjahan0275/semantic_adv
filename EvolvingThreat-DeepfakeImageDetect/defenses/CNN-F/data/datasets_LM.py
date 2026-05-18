import cv2
import numpy as np
import torch
import torch_dct as dct
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================================================
# -------------- FIXED-SIZE DCT PIPELINE ----------------
# ======================================================
# IMPORTANT:
#   • NO RESIZING
#   • NO MULTI-SCALE
#   • ONLY 8×8 and 16×16 blocks on full resolution
#   • produces FIXED feature vector for all images
# ======================================================

# New blockify started

def blockify(x, block):
    """Split an H×W tensor into blocks of block×block."""
    H = x.shape[-2]
    W = x.shape[-1]

    Hc = (H // block) * block
    Wc = (W // block) * block

    x = x[:, :Hc, :Wc]

    # Create non-overlapping patches
    x = x.unfold(1, block, block).unfold(2, block, block)
    return x.contiguous().view(-1, block * block)

# New blockify finished


def extract_stats(v):
    """Return mean, var, skewness, kurtosis."""
    m = v.mean(dim=1)
    s = v.std(dim=1) + 1e-8
    z = (v - m[:, None]) / s[:, None]
    return torch.stack([
        m,
        v.var(dim=1),
        (z ** 3).mean(dim=1),
        (z ** 4).mean(dim=1)
    ], dim=1)  # B, 4


def oriented_energy(coeffs, H, W):
    """Compute simple orientation energies via gradients."""
    gx = coeffs[:, 1:, :] - coeffs[:, :-1, :]
    gy = coeffs[:, :, 1:] - coeffs[:, :, :-1]

    gx = gx.reshape(-1)
    gy = gy.reshape(-1)

    return torch.tensor([
        gx.abs().mean(),
        gy.abs().mean(),
        (gx ** 2).mean(),
        (gy ** 2).mean(),
    ], device=coeffs.device)

def make_patches(img, patch=224, stride=224): 
    """Extract non-overlapping 224x224 patches from full-res image."""
    w, h = img.size

    patches = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            crop = img.crop((x, y, x+patch, y+patch))
            patches.append(crop)

    return patches  # list of PIL images

# New extract_dct_stats started

def extract_dct_stats(img_tensor):
    gray = img_tensor.mean(dim=0, keepdim=True)
    d = torch.abs(dct.dct_2d(gray))

    block_sizes = [8, 16]
    feats = []

    for B in block_sizes:
        blocks = blockify(d, B)
        stats = extract_stats(blocks)
        feats.append(stats.reshape(-1))

    feats = torch.cat(feats)
    feats = (feats - feats.mean()) / (feats.std() + 1e-8)
    return feats  # FINAL FIXED VECTOR: 81920 dims

# New extract_dct_stats finished


# -----------------------------
# Dataset selector
# -----------------------------
def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return BinaryHybridDataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode must be "binary" or "filename".')

# -----------------------------
# Hybrid Dataset: image + DCT
# -----------------------------
class BinaryHybridDataset(datasets.ImageFolder):
    def __init__(self, opt, root):
        super().__init__(root)
        self.opt = opt

        # New code started
        
        # Disable resize/crop for full-res DCT
        self.resize_func = transforms.Lambda(lambda x: x)
        self.crop_func   = transforms.Lambda(lambda x: x)
        self.flip_func   = transforms.Lambda(lambda x: x)
        
        # New code finished
    def __getitem__(self, index):
        path, label = self.samples[index]

        # Load once
        img = Image.open(path).convert('RGB')

        # Data augmentation
        img_aug = data_augment(img, self.opt)
        # Old Fixed number of Patches
        # CNN patches
        '''
        patches = make_patches(img_aug, patch=224, stride=224)

        patch_tensors = []
        for p in patches:
            t = transforms.ToTensor()(p)
            t = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(t)
            patch_tensors.append(t)

        patch_batch = torch.stack(patch_tensors)  # [16,3,224,224]
        '''
        ####### New Random Patch-Sampling Code start###########
        # CNN patches
        patches = make_patches(img_aug, patch=224, stride=224)

        # ---- NEW: sample only K patches per image ----
        K = getattr(self.opt, "num_patches", 16)  # default = 16
        if len(patches) > K:
            idx = torch.randperm(len(patches))[:K]
            patches = [patches[i] for i in idx]
        # ----------------------------------------------

        patch_tensors = []
        for p in patches:
            t = transforms.ToTensor()(p)
            t = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(t)
            patch_tensors.append(t)

        patch_batch = torch.stack(patch_tensors)    # [K,3,224,224]
        ########## New Random Patch-Sampling Code finish########
        # DCT computed ONCE
        img_tensor = transforms.ToTensor()(img)
        dct_feat = extract_dct_stats(img_tensor)

        return patch_batch, dct_feat, torch.tensor(label).float()


# -----------------------------
# FileNameDataset (unchanged)
# -----------------------------
class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path

# -----------------------------
# Augmentation functions
# -----------------------------
def data_augment(img, opt):
    img = np.array(img)
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    # enforce fixed size
    h, w = img.shape[:2]
    img = cv2.resize(img, (w, h))

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    elif len(s) == 2:
        return random() * (s[1]-s[0]) + s[0]
    raise ValueError("s must have length 1 or 2.")

def sample_discrete(s):
    return s[0] if len(s)==1 else choice(s)

def gaussian_blur(img, sigma):
    for c in range(3):
        gaussian_filter(img[:,:,c], output=img[:,:,c], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    return jpeg_dict[key](img, compress_val)

rz_dict = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
    'nearest': InterpolationMode.NEAREST
}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
