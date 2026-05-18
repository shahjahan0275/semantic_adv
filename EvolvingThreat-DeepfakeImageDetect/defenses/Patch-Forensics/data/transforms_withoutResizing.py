import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import logging
import PIL.Image
import numpy as np


# Version 2
def get_transform(opt, for_val=False):
    transform_list = []

    if for_val:
        # 🚫 REMOVE Resize and CenterCrop entirely

        # keep test-time augmentations
        if hasattr(opt, 'test_flip') and opt.test_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
        if hasattr(opt, 'test_compression') and opt.test_compression:
            transform_list.append(JPEGCompression(opt.compression))
        if hasattr(opt, 'test_blur') and opt.test_blur:
            transform_list.append(Blur(opt.blur))
        if hasattr(opt, 'test_gamma') and opt.test_gamma:
            transform_list.append(Gamma(opt.gamma))

    else:
        # 🚫 REMOVE RandomResizedCrop / Resize / RandomCrop / CenterCrop

        # keep CNN-specific augmentations
        if opt.cnn_detection_augment:
            transform_list.append(CNNDetectionAugmentations(
                opt.cnn_detection_augment))

        if opt.color_augment:
            transform_list.append(ColorAugmentations())

        if opt.all_augment:
            transform_list.append(AllAugmentations())

        # Patch shuffling (local artifact forcing)
        if hasattr(opt, 'patch_shuffle') and opt.patch_shuffle:
            transform_list.append(
                PatchShuffle(
                    patch_size=opt.patch_size,
                    p=opt.patch_shuffle_prob
                )
            )

        # keep horizontal flip
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # keep tensor conversion & normalization
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ))

    transform = transforms.Compose(transform_list)
    print(transform)
    logging.info(transform)
    return transform


### additional augmentations ### 

class AllAugmentations(object):
    def __init__(self):
        import albumentations
        self.transform = albumentations.Compose([
            albumentations.Blur(blur_limit=3),
            albumentations.JpegCompression(quality_lower=30, quality_upper=100, p=0.5),
            albumentations.RandomBrightnessContrast(),
            albumentations.RandomGamma(gamma_limit=(80, 120)),
            albumentations.CLAHE(),
        ])

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

class CNNDetectionAugmentations(object):
    def __init__(self, prob=0.5):
        import albumentations
        self.transform = albumentations.Compose([
            albumentations.Blur(blur_limit=3, p=prob),
            albumentations.JpegCompression(quality_lower=30, quality_upper=100, p=prob),
        ])
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

class JPEGCompression(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.augmentations.transforms.JpegCompression(p=1)

    def __call__(self, image):
        image_np = np.array(image)
        image_out = self.transform.apply(image_np, quality=self.level)
        image_pil = PIL.Image.fromarray(image_out)
        return image_pil

class Blur(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.Blur(blur_limit=(self.level, self.level), always_apply=True)

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

class Gamma(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.augmentations.transforms.RandomGamma(p=1)

    def __call__(self, image):
        image_np = np.array(image)
        image_out = self.transform.apply(image_np, gamma=self.level/100)
        image_pil = PIL.Image.fromarray(image_out)
        return image_pil

class ColorAugmentations(object):
    def __init__(self):
        import albumentations
        self.transform = albumentations.Compose([
            albumentations.RandomBrightnessContrast(),
            albumentations.RandomGamma(gamma_limit=(80, 120)),
            albumentations.CLAHE(),
        ])

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

# With patch shuffle
'''
class PatchShuffle(object):
    """
    Divide image into patches and randomly shuffle them.
    Works on PIL images.
    """

    def __init__(self, patch_size=16, p=0.5):
        self.patch_size = patch_size
        self.p = p

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image

        img = np.array(image)
        H, W, C = img.shape
        ps = self.patch_size

        # ensure divisibility
        Hc = (H // ps) * ps
        Wc = (W // ps) * ps
        img = img[:Hc, :Wc]

        # extract patches
        patches = []
        for i in range(0, Hc, ps):
            for j in range(0, Wc, ps):
                patches.append(img[i:i+ps, j:j+ps])

        patches = np.array(patches)
        np.random.shuffle(patches)

        # reconstruct image
        out = np.zeros_like(img)
        idx = 0
        for i in range(0, Hc, ps):
            for j in range(0, Wc, ps):
                out[i:i+ps, j:j+ps] = patches[idx]
                idx += 1

        return PIL.Image.fromarray(out)
'''
#  Without Patch Shuffle
class PatchShuffle(object):
    """
    Divide image into patches WITHOUT shuffling.
    This enforces block structure while preserving spatial order.
    """

    def __init__(self, patch_size=16, p=0.5):
        self.patch_size = patch_size
        self.p = p

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image

        img = np.array(image)
        H, W, C = img.shape
        ps = self.patch_size

        # ensure divisibility
        Hc = (H // ps) * ps
        Wc = (W // ps) * ps
        img = img[:Hc, :Wc]

        # extract patches in FIXED order
        patches = []
        for i in range(0, Hc, ps):
            for j in range(0, Wc, ps):
                patches.append(img[i:i+ps, j:j+ps])

        patches = np.array(patches)  # shape: [N, ps, ps, C]

        # 🔒 NO SHUFFLING HERE

        # reconstruct image (same order)
        out = np.zeros_like(img)
        idx = 0
        for i in range(0, Hc, ps):
            for j in range(0, Wc, ps):
                out[i:i+ps, j:j+ps] = patches[idx]
                idx += 1

        return PIL.Image.fromarray(out)
