import os.path
import torch.utils.data as data
from .dataset_util import make_dataset
from PIL import Image
import numpy as np
import torch
from . import transforms
import random
import scipy.fftpack

def compute_dct_stats(img_tensor):
    """
    img_tensor: torch.Tensor [3, H, W] (CPU)
    returns: torch.Tensor [12] (mean,var,skew,kurt × RGB)
    """
    x = img_tensor.numpy()  # [3,H,W]

    # DCT over H and W
    dct = scipy.fftpack.dct(x, axis=1, norm='ortho')
    dct = scipy.fftpack.dct(dct, axis=2, norm='ortho')

    mean = dct.mean(axis=(1, 2))
    var = dct.var(axis=(1, 2))
    std = np.sqrt(var + 1e-8)

    skew = ((dct - mean[:, None, None]) ** 3).mean(axis=(1, 2)) / (std ** 3 + 1e-8)
    kurt = ((dct - mean[:, None, None]) ** 4).mean(axis=(1, 2)) / (std ** 4 + 1e-8)

    stats = np.concatenate([mean, var, skew, kurt], axis=0)  # [12]
    return torch.from_numpy(stats).float()

class PairedDataset(data.Dataset):
    """A dataset class for paired images
    e.g. corresponding real and manipulated images
    """

    def __init__(self, opt, im_path_real, im_path_fake, is_val=False):
        """Initialize this dataset class.

        Parameters:
            opt -- experiment options
            im_path_real -- path to folder of real images
            im_path_fake -- path to folder of fake images
            is_val -- is this training or validation? used to determine
            transform
        """
        super().__init__()
        self.dir_real = im_path_real
        self.dir_fake = im_path_fake

        # if pairs are named in the same order 
        # e.g. real/train/face1.png, real/train/face2.png ...
        #      fake/train/face1.png, fake/train/face2.png ...
        # then this will align them in a batch unless
        # --no_serial_batches is specified
        self.real_paths = sorted(make_dataset(self.dir_real,
                                              opt.max_dataset_size))
        self.fake_paths = sorted(make_dataset(self.dir_fake,
                                              opt.max_dataset_size))
        self.real_size = len(self.real_paths)
        self.fake_size = len(self.fake_paths)
        self.transform = transforms.get_transform(opt, for_val=is_val)
        self.opt = opt

    def __getitem__(self, index):
        real_path = self.real_paths[index % self.real_size]

        if self.opt.no_serial_batches:
            index_fake = random.randint(0, self.fake_size - 1)
        else:
            index_fake = index % self.fake_size

        fake_path = self.fake_paths[index_fake]

        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')

        # image → tensor
        real = self.transform(real_img)
        fake = self.transform(fake_img)

        # 🔥 DCT STATS (CPU, ONCE)
        real_dct = compute_dct_stats(real.cpu())
        fake_dct = compute_dct_stats(fake.cpu())

        return {
            'original': real,
            'manipulated': fake,
            'original_dct': real_dct,
            'manipulated_dct': fake_dct,
            'path_original': real_path,
            'path_manipulated': fake_path
        }


    def __len__(self):
        return max(self.real_size, self.fake_size)
