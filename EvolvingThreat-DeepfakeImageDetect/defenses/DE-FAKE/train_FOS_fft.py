#FFT-based 4th-order spectral statistics
from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F
import torchvision.transforms as transforms
from clipdatasets import real,fakereal
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
from torch import nn 
import argparse
import time
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch_dct as dct

def compute_multiband_fft(img):
    """
    img: (3, 224, 224) tensor
    Returns 36-dim (or 48-dim if 4 bands) FFT multi-band 4th-order stats.
    """

    c, h, w = img.shape

    # Compute FFT magnitude
    fft = torch.fft.fft2(img)
    fft = torch.abs(fft)

    # Frequencies from 0 → min(h,w)
    max_f = min(h, w)

    # ------- choose either 3-band or 4-band --------
    # SAME BAND SIZES AS DCT VERSION
    LOW = (0, 32)        # low frequencies
    MID = (32, 96)       # mid frequencies
    HIGH = (96, 160)     # high frequencies
    # ULTRA = (160, 224) # optional 4th band if needed

    bands = [LOW, MID, HIGH]        # 3 bands → 36 features
    # bands = [LOW, MID, HIGH, ULTRA]   # 4 bands → 48 features

    feats = []

    for (f1, f2) in bands:
        # Protect overflow
        f2 = min(f2, max_f)

        band = fft[:, f1:f2, f1:f2]
        flat = band.reshape(c, -1)

        mean = flat.mean(dim=1)
        var = flat.var(dim=1)

        std = torch.sqrt(var + 1e-8)
        z = (flat - mean[:, None]) / std[:, None]

        skew = torch.mean(z**3, dim=1)
        kurt = torch.mean(z**4, dim=1)

        stats = torch.stack([mean, var, skew, kurt], dim=1)
        feats.append(stats)

    feats = torch.cat(feats, dim=1).reshape(-1)

    # Normalize entire feature vector
    feats = (feats - feats.mean()) / (feats.std() + 1e-6)

    return feats



parser = argparse.ArgumentParser(description='DE-FAKE finetuning')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--inputpath_linear', type=str, default=None, help='path to pretrained linear model')
parser.add_argument('--inputpath_clip', type=str, default=None, help='path to pretrained CLIP model')
parser.add_argument('--outputpath_linear', type=str, default=None, help='path to save linear model - should be like filename.pt')
parser.add_argument('--outputpath_clip', type=str, default=None, help='path to save CLIP model - should be like filename.pt')

args = parser.parse_args()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 
size = 224 


class CustomRealDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):  
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor()
        ]) 
        self.data = pd.read_csv(datafile)  
        self.len = len(self.data.index) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        caption = self.data.iloc[idx]["caption"]
        caption = str(caption)
        label = 0

        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image).float()
        #return image, label, caption
        fft_feat = compute_multiband_fft(image)
        return image, label, "", fft_feat



class CustomFakeDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):  
        self.transform = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor()
        ]) 
        self.data = pd.read_csv(datafile)  
        self.len = len(self.data.index) 

    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        caption = self.data.iloc[idx]["caption"]
        caption = str(caption)
        label = 1

        imgpath = self.data.iloc[idx]["imagepath"]
        image = Image.open(imgpath).convert('RGB')
        image = self.transform(image).float()
        #return image, label, caption
        fft_feat = compute_multiband_fft(image)
        return image, label, "", fft_feat
 


real_train = CustomRealDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/train_real.csv")
real_val = CustomRealDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/val_real.csv")
real_test = CustomRealDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/test_real.csv")

fake_train = CustomFakeDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/train_fake.csv")
fake_val = CustomFakeDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/val_fake.csv")
fake_test = CustomFakeDataset(datafile="/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DE-FAKE/csv_outputs/test_fake.csv")

train_dataset = torch.utils.data.ConcatDataset([real_train, fake_train])
val_dataset = torch.utils.data.ConcatDataset([real_val, fake_val])
test_dataset = torch.utils.data.ConcatDataset([real_test, fake_test])

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )


#linear = NeuralNet(1024,[512,256],2).to(device)
#linear = torch.load(args.inputpath_linear).to(device)
#model = torch.load(args.inputpath_clip).to(device)

# --------------------------------------
# CORRECT: Initialize NEW classifier for image+text (1024 dim)
# --------------------------------------
linear = NeuralNet(
    #input_size=1024,
    input_size=548,
    hidden_size_list=[512, 256],
    num_classes=2
).to(device)

# --------------------------------------
# LOAD CLIP STATE_DICT SAFELY
# --------------------------------------
if args.inputpath_clip is not None:
    state = torch.load(args.inputpath_clip, map_location=device)
    if isinstance(state, dict):
        # Either full CLIP state_dict or wrapped inside "model" key
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    else:
        print("ERROR: The file contains a full CLIP model, not a state_dict.")
        print("You MUST load using: torch.save(model.state_dict(), PATH)")
        exit()


model.eval()

criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(list(linear.parameters())+list(model.parameters()), lr=args.lr)
for p in model.parameters():
    p.requires_grad = False
# UNFREEZE last 2 transformer blocks
for p in model.visual.transformer.resblocks[-1].parameters():
    p.requires_grad = True
for p in model.visual.transformer.resblocks[-2].parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    list(linear.parameters()) +
    list(model.visual.transformer.resblocks[-1].parameters()) +
    list(model.visual.transformer.resblocks[-2].parameters()),
    lr=args.lr,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=True
)



for i in range(args.epoch): 
    loss_epoch = 0
    train_acc = []
    train_true = []
    
    test_acc = []
    test_true = []
    

    for step, (x, y, _, d) in enumerate(tqdm(train_loader)):
        x = x.cuda()
        y = y.cuda()
        d = d.cuda()
        linear.train()
        model.eval() 
        #text = clip.tokenize(list(t), context_length=77, truncate=True).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(x)
            #text_emb = model.encode_text(text)
        # stronger regularization
        imga_embedding = F.dropout(imga_embedding, p=0.2, training=linear.training)
        dct_feat = F.dropout(d.float(), p=0.1, training=linear.training)
        emb = torch.cat((imga_embedding, dct_feat), 1)

        output = linear(emb.float())
        optimizer.zero_grad()
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        predict = output.argmax(1)
        predict = predict.cpu().numpy()
        predict = list(predict)
        train_acc.extend(predict)
        
        y = y.cpu().numpy()
        y = list(y)
        train_true.extend(y)
        
    for step, (x, y, _, d) in enumerate(tqdm(val_loader)):

        x = x.cuda()
        y = y.cuda()
        d = d.cuda()
        model.eval()
        linear.eval()
        #text = clip.tokenize(list(t), context_length=77, truncate=True).to(device)
        with torch.no_grad():
            imga_embedding = model.encode_image(x)
            #text_emb = model.encode_text(text)
        # stronger regularization
        imga_embedding = F.dropout(imga_embedding, p=0.2, training=linear.training)
        
        dct_feat = F.dropout(d.float(), p=0.1, training=linear.training)
        emb = torch.cat((imga_embedding, dct_feat), 1)

        output = linear(emb.float())
        predict = output.argmax(1)
        predict = predict.cpu().numpy()
        predict = list(predict)
        test_acc.extend(predict)
        
        y = y.cpu().numpy()
        y = list(y)
        test_true.extend(y)
    
    print('train')
    print(accuracy_score(train_true,train_acc)) 
    print('validation')
    print(accuracy_score(test_true,test_acc)) 
    
    val_acc = accuracy_score(test_true,test_acc)
    scheduler.step(val_acc)

    # save model
    #torch.save(linear, args.outputpath_linear)
    #torch.save(model, args.outputpath_clip)
    torch.save(linear.state_dict(), args.outputpath_linear)
    torch.save(model.state_dict(), args.outputpath_clip)




