import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from ast import arg
import os
import csv
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
#from models.clip_models_speed import CLIPModelShuffleAttentionPenultimateLayer
from models.clip_models_FOS import CLIPModelShuffleAttentionPenultimateLayer
#from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer
import pandas as pd
from calculate_global_ap import cal_global_ap


# ======= FORCE CPU ========
#device = torch.device("cpu")  # everything runs on CPU
#torch.set_num_threads(os.cpu_count())  # use all CPU cores

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)

# Perturbation Functions to simulate compression, cropping, and screen-captures
def proportional_center_crop(img, crop_pct):
    """Crops the center of the image to a specific percentage of its original size."""
    w, h = img.size
    new_w, new_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    return img.crop((left, top, right, bottom))

def simulate_screenshot(img):
    """
    Simulates a screenshot by introducing screen-fit scaling and standard OS screenshot compression.
    """
    w, h = img.size
    
    # Handle older PIL versions safely
    resample_method = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    
    # Simulate screen viewport (downscaling slightly as if fitting to a monitor)
    img_screen = img.resize((int(w * 0.85), int(h * 0.85)), resample_method)
    
    # Simulate standard OS screenshot capture quantization (often saved as mildly compressed JPEG/PNG)
    out = BytesIO()
    img_screen.save(out, format='jpeg', quality=85)
    img_screen = Image.open(out)
    img_arr = np.array(img_screen)
    out.close()
    
    return Image.fromarray(img_arr)

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False, data_augment=None, threshold=0.5):
    drop_out_rate = 0.0
    shuffle_rate = 0.0
    if hasattr(model, 'drop_out_rate') and model.drop_out_rate != 0.0:
        drop_out_rate = model.drop_out_rate
        model.drop_out_rate = 0.0
    if hasattr(model, 'shuffle_rate') and model.shuffle_rate != 0.0:
        shuffle_rate = model.shuffle_rate
        model.shuffle_rate = 0.0

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.cuda()
            #in_tens = img.to("cpu")
            #in_tens = img.to(device)
            if data_augment != None:
                in_tens = data_augment(in_tens)
            y_pred_temp = model(in_tens)
            if y_pred_temp.shape[-1] == 2:
                # anomaly
                # y_pred_temp = torch.absolute(y_pred_temp[:, 1] - y_pred_temp[:, 0])
                y_pred_temp = y_pred_temp[:, 0]
            y_pred.extend(y_pred_temp.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    if hasattr(model, 'drop_out_rate'):
        model.drop_out_rate = drop_out_rate
    if hasattr(model, 'shuffle_rate'):
        model.shuffle_rate = shuffle_rate
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true



def decoupled_validate(model, loader, find_thres=False, data_augment=None):
    
    with torch.no_grad():
        y_true, y_orig_pred, y_shuffle_pred = [], [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.cuda()
            #in_tens = img.to("cpu")
            #in_tens = img.to(device)
            if data_augment != None:
                in_tens = data_augment(in_tens)
            #model.set_input((in_tens, label.cuda()))
            #model.set_input((in_tens, label.to("cpu")))
            #model.set_input((in_tens, label.to(device)))
            #model.decoupled_input_forward()
            model_output = model.output
            orig_output = model_output[:, 1]
            shuffle_output = model_output[:, 2]
            y_orig_pred.extend(orig_output.sigmoid().flatten().tolist())
            y_shuffle_pred.extend(shuffle_output.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_orig_pred, y_shuffle_pred = np.array(y_true), np.array(y_orig_pred), np.array(y_shuffle_pred)
    avg_pred = np.array([(y_orig_pred[i]+y_shuffle_pred[i])/2 for i in range(len(y_orig_pred))])
    pred = [y_orig_pred, y_shuffle_pred, avg_pred]

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = [average_precision_score(y_true, p) for p in pred]


    # Acc based on 0.5
    acc = [calculate_acc(y_true, p, 0.5)[2] for p in pred]
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    # r_acc0, f_acc0, orig_acc0 = calculate_acc(y_true, y_orig_pred, 0.5)
    if not find_thres:
        return ap, acc


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_orig_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_orig_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

    
    



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list




# Original without compression, cropping, and screen-captures
'''
class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        is_norm=True):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        if is_norm:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])


    def read_path(self, real_path, fake_path, max_sample):

        real_list = get_list(real_path, must_contain='')
        fake_list = get_list(fake_path, must_contain='')



        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        else:
            max_sample = min(len(fake_list), len(real_list))
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        assert len(real_list) == len(fake_list)  

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label
'''
# with compression, cropping, and screen-captures
class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        crop_pct=None,           # <--- NEW
                        is_screenshot=False,     # <--- NEW
                        is_norm=True):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.crop_pct = crop_pct                 # <--- NEW
        self.is_screenshot = is_screenshot       # <--- NEW
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        if is_norm:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])


    def read_path(self, real_path, fake_path, max_sample):
        # ... (Keep your existing read_path method exactly the same) ...
        real_list = get_list(real_path, must_contain='')
        fake_list = get_list(fake_path, must_contain='')

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        else:
            max_sample = min(len(fake_list), len(real_list))
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        assert len(real_list) == len(fake_list)  

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        # --- APPLY PERTURBATIONS BEFORE TENSOR TRANSFORMS ---
        if self.is_screenshot:
            img = simulate_screenshot(img)
            
        if self.crop_pct is not None:
            img = proportional_center_crop(img, self.crop_pct)
            
        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
            
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label

# Modification 3 with individual image prediction with dynamic best threshold search
# assuming you already defined RealFakeDataset, CLIPModelShuffleAttentionPenultimateLayer, set_seed, etc.
# Without compression, cropping, and screen-captures
'''
if __name__ == '__main__':

    # basic configuration
    set_seed(418)
    result_folder = "/forcolab/home/ashahj/D3/results"
    #checkpoint = "/forcolab/home/ashahj/D3/ckpt/original/classifier.pth"
    #checkpoint = "/speed-scratch/a_shahj/D3/ckpt/train_d3B_DCT/model_epoch_0.pth"
    checkpoint = "/forcolab/home/ashahj/D3/ckpt/train_d3B_FO4_DCT_StyleCLIP_ReRUN/model_epoch_best.pth"
    #checkpoint = "/forcolab/home/ashahj/D3/ckpt/train_d3B_FO4_DCT_StyleCLIP_genimage_ReRUN/model_epoch_best.pth"
    #checkpoint = "/forcolab/home/ashahj/D3/ckpt/train_d3B_FO4_DCT_StyleCLIP_genimage_ReRUN/model_epoch_45.pth"
    # dataset paths
    #real_path = "/forcolab/home/ashahj/D3/data/surrogate_StyleCLIP_dataset/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/surrogate_StyleCLIP_dataset/1_fake"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN-F_VIGTEXT_V2/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CLIPResNet/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CLIPResNet/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/EfficientNet/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/EfficientNet/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/ViT/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/ViT/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/advV2_2000/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/advV2_2000/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/ForDefenses/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/ForDefenses/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/ForSurrogate/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/ForSurrogate/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/deepfake/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/deepfake/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/whichfaceisreal/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/whichfaceisreal/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/biggan/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/biggan/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/crn/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/crn/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/gaugan/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/gaugan/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/imle/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/imle/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/san/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/san/1_fake"
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/seeingdark/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/seeingdark/1_fake"
    real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/stargan/0_real"
    fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/stargan/1_fake"


    batch_size = 4
    jpeg_quality = None
    gaussian_sigma = None
    exp_name = f"validation"
    max_sample = None

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create and load the detector
    granularity = 14
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[granularity]
    )
    is_norm = True
    print(f"using checkpoint {checkpoint}")
    state_dict = torch.load(checkpoint, map_location='cpu')

    ckpt = torch.load(checkpoint, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    # 🔥 REMOVE DCT BUFFER (dynamic, rebuild at runtime)
    sd = {
        k: v for k, v in sd.items()
        if not k.startswith("dct_moment_pool.dct_mat")
    }

    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    print("ℹ️ Missing keys:", missing)
    print("ℹ️ Unexpected keys:", unexpected)

    print("Model loaded..")
    model.eval()
    model.cuda()
    #model.to(device)
    #model.to("cpu")

    arch = "clip"

    # dataset
    dataset = RealFakeDataset(
        real_path,
        fake_path,
        max_sample,
        arch,
        jpeg_quality=jpeg_quality,
        gaussian_sigma=gaussian_sigma,
        is_norm=is_norm
    )

    print("Length of dataset:", len(dataset))

    if len(dataset) == 0:
        raise ValueError("❌ Dataset is empty! Check dataset paths or file formats.")
'''
# With compression, cropping, and screen-captures
if __name__ == '__main__':

    # basic configuration
    set_seed(418)
    result_folder = "/forcolab/home/ashahj/D3/results"
    checkpoint = "/forcolab/home/ashahj/D3/ckpt/train_d3B_FO4_DCT_StyleCLIP_ReRUN/model_epoch_best.pth"
    
    #real_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/stargan/0_real"
    #fake_path = "/forcolab/home/ashahj/D3/data/CNN_synth_testset/stargan/1_fake"
    real_path = "/forcolab/home/ashahj/D3/data/surrogate_StyleCLIP_dataset/0_real"
    fake_path = "/forcolab/home/ashahj/D3/data/surrogate_StyleCLIP_dataset/1_fake"

    batch_size = 4
    max_sample = None
    
    # ==========================================================
    # 🧪 ROBUSTNESS TESTING TOGGLES
    # Activate ONE of the following tests by setting its value. 
    # Keep the others set to None/False for isolated testing.
    # ==========================================================
    
    TEST_JPEG_QUALITY = None   # Example: Set to 50 for heavy compression, 75 for medium
    TEST_CROP_PCT = None       # Example: Set to 0.8 to crop the central 80% of the image
    TEST_SCREENSHOT = True    # Example: Set to True to simulate screen recording/captures
    
    # Set up dynamic experiment naming to avoid overwriting Excel files
    exp_name = "validation_baseline"
    if TEST_JPEG_QUALITY is not None:
        exp_name = f"validation_jpeg_{TEST_JPEG_QUALITY}"
    elif TEST_CROP_PCT is not None:
        exp_name = f"validation_crop_{int(TEST_CROP_PCT*100)}pct"
    elif TEST_SCREENSHOT:
        exp_name = "validation_screenshot"
        
    print(f"Running Experiment: {exp_name}")
    # ==========================================================

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create and load the detector
    granularity = 14
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[granularity]
    )
    is_norm = True
    print(f"using checkpoint {checkpoint}")
    
    ckpt = torch.load(checkpoint, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
        
    # 🔥 REMOVE DCT BUFFER (dynamic, rebuild at runtime)
    sd = {
        k: v for k, v in sd.items()
        if not k.startswith("dct_moment_pool.dct_mat")
    }

    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    print("ℹ️ Missing keys:", missing)
    print("ℹ️ Unexpected keys:", unexpected)

    print("Model loaded..")
    model.eval()
    model.cuda()

    arch = "clip"

    # dataset instantiation with the new parameters passed in
    dataset = RealFakeDataset(
        real_path,
        fake_path,
        max_sample,
        arch,
        jpeg_quality=TEST_JPEG_QUALITY,
        gaussian_sigma=None,
        crop_pct=TEST_CROP_PCT,         # Passed from toggles
        is_screenshot=TEST_SCREENSHOT,  # Passed from toggles
        is_norm=is_norm
    )

    print("Length of dataset:", len(dataset))

    if len(dataset) == 0:
        raise ValueError("❌ Dataset is empty! Check dataset paths or file formats.")


    # ... (Keep the rest of your evaluation logic exactly the same from here down) ...
    # ✅ store per-image results

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ✅ store per-image results
    all_results = []
    all_labels = []
    all_fake_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            #imgs = imgs.to(device)
            #imgs = imgs.to("cpu")
            outputs = model(imgs)  # shape [B, 1]

            # convert logits -> probability of "fake"
            #fake_probs = torch.sigmoid(outputs).squeeze(dim=1).cpu().numpy()
            outputs = outputs.view(outputs.size(0), -1)  # [B,1] guaranteed
            fake_probs = torch.sigmoid(outputs[:, 0]).cpu().numpy()

            real_probs = 1.0 - fake_probs

            labels = labels.cpu().numpy()
            preds = (fake_probs > 0.5).astype(int)  # default 0.5 preds (used only for table)

            # accumulate for metrics
            all_labels.extend(labels)
            all_fake_probs.extend(fake_probs)

            # get file names from dataset
            batch_indices = range(len(all_results), len(all_results) + len(labels))
            file_names = [dataset.total_list[i] for i in batch_indices]

            for fname, truth, pred, fs, rs in zip(file_names, labels, preds, fake_probs, real_probs):
                all_results.append({
                    "image_name": os.path.basename(fname),
                    "truth": int(truth),   # 0 = real, 1 = fake
                    "pred": int(pred),
                    "fake_score": float(fs),
                    "real_score": float(rs)
                })

    # ----------------------------
    # ✅ Compute global metrics with best threshold
    # ----------------------------
    all_labels = torch.tensor(all_labels).numpy()
    all_fake_probs = torch.tensor(all_fake_probs).numpy()

    ap = average_precision_score(all_labels, all_fake_probs)

    # sweep thresholds to find best balanced accuracy
    thresholds = torch.linspace(0, 1, 101).numpy()
    best_bacc, best_acc, best_thr = 0, 0, 0.5
    for thr in thresholds:
        preds_thr = (all_fake_probs > thr).astype(int)
        bacc = balanced_accuracy_score(all_labels, preds_thr)
        acc = accuracy_score(all_labels, preds_thr)
        if bacc > best_bacc:
            best_bacc = bacc
            best_acc = acc
            best_thr = thr
    
    # ✅ Use best threshold to compute extra metrics
    best_preds = (all_fake_probs > best_thr).astype(int)

    f1 = f1_score(all_labels, best_preds)
    precision = precision_score(all_labels, best_preds)
    recall = recall_score(all_labels, best_preds)

    tn, fp, fn, tp = confusion_matrix(all_labels, best_preds).ravel()


    print(f"📊 Best Threshold = {best_thr:.2f}, Accuracy = {best_acc:.4f}, "
          f"AP = {ap:.4f}, Balanced Acc = {best_bacc:.4f},"
          f"F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # ----------------------------
    # ✅ Save to Excel (metrics + per-image results)
    # ----------------------------
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(result_folder, f"{exp_name}_per_image.xlsx")

    with pd.ExcelWriter(results_path, engine="openpyxl") as writer:
        # Save metrics on top
        metrics_df = pd.DataFrame([{
            "Best Threshold": best_thr,
            "Accuracy": best_acc,
            "AP": ap,
            "Balanced Accuracy": best_bacc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }])
        metrics_df.to_excel(writer, sheet_name="results", index=False, startrow=0)

        # Save per-image results below
        results_df.to_excel(writer, sheet_name="results", index=False, startrow=4)

    print(f"✅ Saved metrics + per-image results to {results_path}")
    print(results_df.head(10))


