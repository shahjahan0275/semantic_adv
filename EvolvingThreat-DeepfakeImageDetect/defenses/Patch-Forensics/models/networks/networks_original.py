import torch
import torch.nn as nn
import torch.fft as fft
import scipy.fftpack
#from IPython import embed
from . import netutils

# Version 1
'''
class MultiBandDCTStats(nn.Module):
    def __init__(self, bands=((0,8),(8,16),(16,32))):
        super().__init__()
        self.bands = bands

    def dct2(self, x):
        x = fft.dct(x, norm='ortho', dim=-1)
        x = fft.dct(x, norm='ortho', dim=-2)
        return x

    def compute_stats(self, x):
        mean = x.mean(dim=[2,3])
        var  = x.var(dim=[2,3], unbiased=False)
        std  = torch.sqrt(var + 1e-6)

        skew = ((x - mean[:,:,None,None])**3).mean(dim=[2,3]) / (std**3 + 1e-6)
        kurt = ((x - mean[:,:,None,None])**4).mean(dim=[2,3]) / (std**4 + 1e-6)

        return torch.cat([mean, var, skew, kurt], dim=1)

    def forward(self, x):
        dct = self.dct2(x)
        feats = []
        for (l, h) in self.bands:
            band = dct[:, :, l:h, l:h]
            feats.append(self.compute_stats(band))
        return torch.cat(feats, dim=1)
'''
# Version 2
class MultiBandDCTStats(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x : [B, C, H, W]
        B, C, H, W = x.shape

        # move to CPU for scipy DCT
        x_np = x.detach().cpu().numpy()

        # 2D DCT (H then W)
        dct_h = scipy.fftpack.dct(x_np, axis=2, norm='ortho')
        dct_hw = scipy.fftpack.dct(dct_h, axis=3, norm='ortho')

        dct = torch.from_numpy(dct_hw).to(x.device)

        # ---- Statistics over spatial dims ----
        mean = dct.mean(dim=(2, 3))
        var  = dct.var(dim=(2, 3), unbiased=False)
        std  = torch.sqrt(var + 1e-8)

        skew = ((dct - mean[:, :, None, None]) ** 3).mean(dim=(2, 3)) / (std ** 3 + 1e-8)
        kurt = ((dct - mean[:, :, None, None]) ** 4).mean(dim=(2, 3)) / (std ** 4 + 1e-8)

        # [B, 4*C]
        stats = torch.cat([mean, var, skew, kurt], dim=1)
        return stats
#Version 1
'''
class ResNetWithDCT(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()

        # Split ResNet
        self.backbone = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
        )

        self.avgpool = resnet_model.avgpool

        # CNN feature dim (ResNet-18/34 = 512, 50+ = 2048)
        cnn_dim = resnet_model.fc.in_features

        # DCT branch
        self.dct_stats = MultiBandDCTStats(
            bands=[(0,8),(8,16),(16,32)]
        )

        dct_dim = 3 * 4 * 3  # bands × stats × channels

        self.fc = nn.Linear(cnn_dim + dct_dim, 2)

    def forward(self, x):
        # CNN path
        feat = self.backbone(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        # DCT path
        dct_feat = self.dct_stats(x)

        # Fuse
        fused = torch.cat([feat, dct_feat], dim=1)
        return self.fc(fused)
'''

# Version 2
class ResNetWithDCT(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()

        self.backbone = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
        )

        self.avgpool = resnet_model.avgpool
        cnn_dim = resnet_model.fc.in_features

        # NEW DCT extractor
        self.dct_stats = MultiBandDCTStats()
        dct_dim = 4 * 3  # mean, var, skew, kurt × RGB

        self.fc = nn.Linear(cnn_dim + dct_dim, 2)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        dct_feat = self.dct_stats(x)

        fused = torch.cat([feat, dct_feat], dim=1)
        return self.fc(fused)


def modify_commandline_options(parser):
    opt, _ = parser.parse_known_args()
    if 'xception' in opt.which_model_netD:
        parser.set_defaults(loadSize=333, fineSize=299)
    elif 'resnet' in opt.which_model_netD:
        parser.set_defaults(loadSize=256, fineSize=224)
    else:
        raise NotImplementedError

def define_D(which_model_netD, init_type, gpu_ids=[]):
    if 'resnet' in which_model_netD:
        from torchvision.models import resnet

        model_fn = getattr(resnet, which_model_netD)
        base_resnet = model_fn(pretrained=False)

        netD = ResNetWithDCT(base_resnet)


    elif 'xception' in which_model_netD:
        from . import xception
        netD = xception.xception(num_classes=2)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)

def define_patch_D(which_model_netD, init_type, gpu_ids=[]):
    if which_model_netD.startswith('resnet'):
        # e.g. which_model_netD = resnet18_layer1
        from . import customnet
        depth = int(which_model_netD.split('_')[0][6:])
        layer = which_model_netD.split('_')[1]
        netD = customnet.make_patch_resnet(depth, layer)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('widenet'):
        # e.g. which_model_netD = widenet_kw7_d1
        splits = which_model_netD.split('_')
        kernel_size = int(splits[1][2:])
        dilation = int(splits[2][1:])
        netD = WideNet(kernel_size, dilation)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('xception'):
        # e.g. which_model_netD = xceptionnet_block2
        from . import customnet
        splits = which_model_netD.split('_')
        layer = splits[1]
        netD = customnet.make_patch_xceptionnet(layer)
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    elif which_model_netD.startswith('longxception'):
        from . import customnet
        netD = customnet.make_xceptionnet_long()
        return netutils.init_net(netD, init_type, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

class WideNet(nn.Module):
    # a shallow network based off initial layers of resnet with 
    # a few 1x1 conv layers added on
    def __init__(self, kernel_size=7, dilation=1):
        super().__init__()
        sequence = [
            nn.Conv2d(3, 256, kernel_size=kernel_size, dilation=dilation,
                      stride=2, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # linear layers
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

