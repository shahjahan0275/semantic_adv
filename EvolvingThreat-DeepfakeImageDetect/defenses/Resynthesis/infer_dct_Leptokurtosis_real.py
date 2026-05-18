import argparse
import os
import random
import shutil
import time
import warnings
import sys
import cv2

import numpy as np
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import scipy.fftpack  # ADD THIS IMPORT at the top
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
#import torchvision.models as models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import get_dataset
from models import get_classification_model

from sr_models.model import RDN, Vgg19
'''
#PyTorch ≥ 2.0
class DCTStatExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        # x : [B, C, H, W]
        dct = torch.fft.dct(torch.fft.dct(x, norm="ortho", dim=2), norm="ortho", dim=3)
        mean = dct.mean(dim=(2,3))
        var  = dct.var(dim=(2,3))
        std = torch.sqrt(var + 1e-8)
        skew = ((dct - mean[:,:,None,None])**3).mean(dim=(2,3)) / (std**3 + 1e-8)
        kurt = ((dct - mean[:,:,None,None])**4).mean(dim=(2,3)) / (std**4 + 1e-8)
        stats = torch.cat([mean, var, skew, kurt], dim=1)
        return stats
'''

class DCTStatExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, 512, H, W]
        device = x.device

        x_np = x.detach().cpu().numpy()

        dct_h  = scipy.fftpack.dct(x_np, axis=2, norm='ortho')
        dct_hw = scipy.fftpack.dct(dct_h, axis=3, norm='ortho')

        dct = torch.from_numpy(dct_hw).to(device)

        mean = dct.mean(dim=(2, 3))
        var  = dct.var(dim=(2, 3))
        std  = torch.sqrt(var + 1e-8)

        skew = ((dct - mean[:, :, None, None]) ** 3).mean(dim=(2, 3)) / (std ** 3 + 1e-8)
        kurt = ((dct - mean[:, :, None, None]) ** 4).mean(dim=(2, 3)) / (std ** 4 + 1e-8)

        return torch.cat([mean, var, skew, kurt], dim=1)  # [B, 2048]


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-root-pos', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--data-root-neg', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cityscapes',
                            help='dataset name (default: pascal12)')
parser.add_argument('-a', '--arch', type=str, default='resnet50',                   
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--input-channel', default=3, type=int,
                    help='number of input channel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save-every-epoch', type=int, default=10,
                    help='how many epochs to save a model.')
parser.add_argument('--output-path', default='./output_models', type=str, metavar='PATH',
                    help='path to output models')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dataset_type', type=str, default='image',
                    help='which dataset to load.')

parser.add_argument('--carlibration', default=1.0, type=float,
                    help='carlibration factor for posterior')
parser.add_argument('--defense', default=1.0, type=float,
                    help='defense factor')
parser.add_argument('--save_path', type=str, default='./score.npy', help='save models')

parser.add_argument('--no_dilation', action='store_true', help='do not use dilated convolutions in attackers')

parser.add_argument('--sr-num-features', type=int, default=64)
parser.add_argument('--sr-growth-rate', type=int, default=64)
parser.add_argument('--sr-num-blocks', type=int, default=16)
parser.add_argument('--sr-num-layers', type=int, default=8)
parser.add_argument('--sr-scale', type=int, default=4)

parser.add_argument('--sr-weights-file', type=str, required=True)

parser.add_argument('--idx-stages', type=int, default=0) 

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = get_classification_model(arch=args.arch, pretrained = args.pretrained,
                                     input_channel=args.input_channel, num_classes=2, dilated=(not args.no_dilation))
    dct_extractor = DCTStatExtractor().cuda(args.gpu)

    #import ipdb; ipdb.set_trace()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:%d'%(args.gpu))
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    test_dataset = get_dataset(name=args.dataset_type, root_pos=args.data_root_pos, root_neg=args.data_root_neg, flip=False)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    sr_model = RDN(scale_factor=args.sr_scale,
                num_channels=3,
                num_features=args.sr_num_features,
                growth_rate=args.sr_growth_rate,
                num_blocks=args.sr_num_blocks,
                num_layers=args.sr_num_layers,
                requires_grad=False).cuda(args.gpu)#.to(device)
    
    checkpoint = torch.load(args.sr_weights_file, map_location='cuda:%d'%(args.gpu))
    if 'state_dict' in checkpoint.keys():
        sr_model.load_state_dict(checkpoint['state_dict'])
    else:
        sr_model.load_state_dict(checkpoint)

    perception_net = Vgg19().cuda(args.gpu)

    Score = test(test_loader, model, sr_model, perception_net, dct_extractor, args)
    np.save(args.save_path, Score)
    # ===== t-SNE FEATURE EXTRACTION =====
    '''
    from extract_tsne_features import extract_features

    X_dct, _ = extract_features(
        test_loader,
        model,
        sr_model,
        perception_net,
        args,
        gpu=args.gpu,
        dct_extractor=dct_extractor
    )

    np.save("tsne_dct.npy", X_dct)
    print("Saved tsne_dct.npy")
    '''

def test(test_loader, model, sr_model, perception_net, dct_extractor, args):
    tp, fp, fn, tn = 0, 0, 0, 0
    model.eval()
    sr_model.eval()
    perception_net.eval()
    
    viz_dir = "visualizations_real"
    os.makedirs(viz_dir, exist_ok=True)
    
    viz_count_fake, viz_count_real, max_viz = 0, 0, 5 

    for i, (input, target, post_path) in enumerate(test_loader):     
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # 1. SR Processing
        lr = 0
        for ii in range(args.sr_scale):
            for jj in range(args.sr_scale):
                lr = lr + input[:, :, ii::args.sr_scale, jj::args.sr_scale] / (args.sr_scale**2)
        lr, input = lr / 255.0, input / 255.0

            
        with torch.no_grad():
            preds_input = sr_model(lr)

            vgg_preds = perception_net(preds_input)
            vgg_input = perception_net(input)

            idx = args.idx_stages - 1 if args.idx_stages > 0 else 3
            rec_features = abs(vgg_preds[idx] - vgg_input[idx])  # [B, 512, H, W]

            residual = abs(preds_input - input)  # [B, 3, H, W]

            # Visualization only
            res_for_viz = torch.clamp(residual.clone() - 0.05, min=0)

            # DCT statistics MUST be computed from rec_features
            dct_stats = dct_extractor(rec_features)

            # 🔒 SAFETY CHECKS (put them HERE)
            assert rec_features.shape[1] == 512, \
                f"Expected 512 feature channels, got {rec_features.shape}"
            assert dct_stats.shape[1] == 2048, \
                f"Expected 2048-D DCT stats, got {dct_stats.shape}"

            # Model inference
            output, _ = model(rec_features, dct_stats)



        # 7. Prediction & Scoring
        target_val = target.cpu().numpy()[0]
        pred_val = (output[:, 0] < output[:, 1]).cpu().numpy()[0]
        
        # Calculate a simple kurtosis for logging/viz purposes only
       
        # Kurtosis from TRUE residual (logging only)
        #res_np = residual.detach().cpu().numpy()
        #k_val = ((res_np - res_np.mean())**4).mean() / (res_np.var()**2 + 1e-8)

        #visualization kurtosis match the model
        # ----- CORRECT kurtosis computation for visualization -----
        # rec_features: [B, 512, H, W]
        rec_np = rec_features.detach().cpu().numpy()

        # 2D DCT
        dct_hw = scipy.fftpack.dct(
            scipy.fftpack.dct(rec_np, axis=2, norm='ortho'),
            axis=3, norm='ortho'
        )

        B, C, H, W = dct_hw.shape
        eps = 1e-8
        kurtosis_vals = []
        low_freq_cut = int(0.1 * min(H, W))  # remove first 10% of H/W as low frequencies

        for c in range(C):
            # High-frequency coefficients only
            coeffs = dct_hw[0, c, low_freq_cut:, low_freq_cut:].reshape(-1)
            
            var = coeffs.var()
            if var < 1e-6:
                continue  # skip flat channels

            # normalize
            coeffs = (coeffs - coeffs.mean()) / (np.sqrt(var) + eps)
            
            # 4th moment = kurtosis
            k = np.mean(coeffs**4)
            if np.isfinite(k) and k < 100:
                kurtosis_vals.append((k, var))  # store variance too

        if len(kurtosis_vals) > 0:
            # weighted mean by channel variance
            ks, vs = zip(*kurtosis_vals)
            k_val = np.average(ks, weights=vs)
        else:
            k_val = 0.0

         
        # 8. Visualization
        is_fake = (target_val == 1)
        if (is_fake and viz_count_fake < max_viz) or (not is_fake and viz_count_real < max_viz):
            if is_fake: viz_count_fake += 1
            else: viz_count_real += 1
            
            viz_np = res_for_viz.detach().cpu().numpy()
            dct_hw = scipy.fftpack.dct(
                scipy.fftpack.dct(viz_np, axis=2, norm='ortho'),
                axis=3, norm='ortho'
            )

            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(viz_np[0], axis=0), cmap='hot', vmin=0, vmax=0.2)

            plt.title(f"{'Fake' if is_fake else 'Real'} Residual\nPred: {'Fake' if pred_val else 'Real'}")
            
            plt.subplot(1, 2, 2)
            plt.hist(dct_hw.flatten(), bins=100, color='red' if is_fake else 'blue', alpha=0.7, log=True)
            plt.title(f"DCT Distribution (Kurtosis: {k_val:.2f})")
            plt.savefig(f"{viz_dir}/{'fake' if is_fake else 'real'}_{i}.png")
            plt.close()

        target_np = target.cpu().numpy()
        pred = (output[:, 0] < output[:, 1]).cpu().numpy()
        tp += sum((target_np == 1) & (pred == 1))
        fp += sum((target_np == 0) & (pred == 1))
        fn += sum((target_np == 1) & (pred == 0))
        tn += sum((target_np == 0) & (pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    print(f"Test Accuracy: {accuracy:.4f} | TP: {tp} TN: {tn} FP: {fp} FN: {fn}")
    return accuracy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()





