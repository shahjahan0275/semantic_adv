# Last modification LM1
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
    # v: [num_blocks, block*block]
    m = v.mean(dim=1)
    var = v.var(dim=1)
    s = v.std(dim=1) + 1e-8
    z = (v - m[:, None]) / s[:, None]

    skew = (z ** 3).mean(dim=1)
    kurt = (z ** 4).mean(dim=1)

    return torch.stack([m, var, skew, kurt], dim=1)  # [num_blocks, 4]


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

def make_patches(img, patch=224):
    w,h = img.size
    
    # zero-pad smaller images
    if w < patch or h < patch:
        new_w = max(w, patch)
        new_h = max(h, patch)
        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)

    w,h = img.size
    patches = []

    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            patches.append(img.crop((x,y,x+patch,y+patch)))

    return patches

# New extract_dct_stats started
def extract_dct_stats(img_tensor):
    # FIXED RESOLUTION FOR DCT ONLY
    img_tensor = TF.resize(img_tensor, (512, 512), interpolation=InterpolationMode.BICUBIC)

    gray = img_tensor.mean(dim=0, keepdim=True)
    d = torch.abs(dct.dct_2d(gray))

    block_sizes = [8,16]
    feats = []

    for B in block_sizes:
        blocks = blockify(d, B)
        stats = extract_stats(blocks)
        feats.append(stats.reshape(-1))

    feats = torch.cat(feats)
    return feats

# New extract_dct_stats finished


# -----------------------------
# Dataset selector
# -----------------------------
def dataset_folder(opt, root, is_train=True):
    if opt.mode == 'binary':
        return BinaryHybridDataset(opt, root, is_train=is_train)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)


# -----------------------------
# Hybrid Dataset: image + DCT
# -----------------------------
class BinaryHybridDataset(datasets.ImageFolder):
    def __init__(self, opt, root, is_train=True):
        super().__init__(root)
        self.opt = opt
        self.is_train = is_train

        # Disable resize/crop for full-res DCT
        self.resize_func = transforms.Lambda(lambda x: x)
        self.crop_func   = transforms.Lambda(lambda x: x)
        self.flip_func   = transforms.Lambda(lambda x: x)
        # NEW: ONLINE RUNNING MEAN / STD INITIALIZATION
        self.dct_mean = None
        self.dct_std  = None
        self.n_seen   = 0

        
        # New code finished
    def __getitem__(self, index):
        path, label = self.samples[index]

        # Load once
        img = Image.open(path).convert('RGB')

        # Data augmentation
        #img_aug = data_augment(img, self.opt)
        if self.is_train:
            img_aug = data_augment(img, self.opt)
        else:
            img_aug = img    # no blur, no jpeg, no resize


        ####### New Random Patch-Sampling Code start###########
        # CNN patches
        patches = make_patches(img_aug, patch=224)

        # ---- NEW: sample only K patches per image ----
        K = getattr(self.opt, "num_patches", 16)  # default = 16
        
        if self.is_train:
            if len(patches) > K:
                idx = torch.randperm(len(patches))[:K]
                patches = [patches[i] for i in idx]
        else:
            # deterministic first K patches
            patches = patches[:K]

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
        img_tensor = transforms.ToTensor()(img_aug)
        dct_feat = extract_dct_stats(img_tensor)

        # -----------------------------------------------------
        # NEW: ONLINE MEAN / STD UPDATE
        # -----------------------------------------------------
        if self.is_train:
            if self.dct_mean is None:
                self.dct_mean = dct_feat.clone()
                self.dct_std = torch.zeros_like(dct_feat)
                self.n_seen = 1
            else:
                self.n_seen += 1
                delta = dct_feat - self.dct_mean
                self.dct_mean += delta / self.n_seen
                delta2 = dct_feat - self.dct_mean
                self.dct_std += delta * delta2

        # Compute std using stored values
        std = torch.sqrt(self.dct_std / max(self.n_seen - 1, 1) + 1e-6)

        # Normalize using TRAINING statistics (fixed)
        dct_feat = (dct_feat - self.dct_mean) / (std + 1e-6)

        # -----------------------------------------------------
        return patch_batch, dct_feat, torch.tensor(label).float()
    
    # ============================================================
    #  EXPOSE DCT MEAN & STD FOR SAVING IN CHECKPOINT
    # ============================================================

    def get_dct_mean(self):
        if self.dct_mean is None:
            raise RuntimeError("DCT mean not computed yet — dataset not iterated.")
        return self.dct_mean.clone()

    def get_dct_std(self):
        if self.dct_std is None:
            raise RuntimeError("DCT std not computed yet — dataset not iterated.")
        return self.dct_std.clone()

    def finalize_stats(self):
        """Finalize the running mean and variance into true std."""
        if self.n_seen < 2:
            return

        # convert accumulated variance into std
        var = self.dct_std / (self.n_seen - 1)
        self.dct_std = torch.sqrt(var + 1e-6)


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
