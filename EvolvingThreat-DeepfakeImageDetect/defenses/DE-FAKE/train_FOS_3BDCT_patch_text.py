# DCT PATCH-BASED 4th order spectral statistics
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


# ---------------------------------------------------------
# PATCH-BASED DCT 4TH-ORDER SPECTRAL STATISTICS
# ---------------------------------------------------------
def compute_patch_dct_stats(img, num_patches=4):
    """
    img: (3, 224, 224)
    Returns 576-dim patchwise DCT 4th-order statistics
    """

    C, H, W = img.shape
    patch_h = H // num_patches
    patch_w = W // num_patches

    LOW = (0, 16)
    MID = (16, 48)
    HIGH = (48, 80)
    bands = [LOW, MID, HIGH]

    features = []


    for py in range(num_patches):
        for px in range(num_patches):

            patch = img[:, py*patch_h:(py+1)*patch_h,
                        px*patch_w:(px+1)*patch_w]

            dct_patch = dct.dct_2d(patch)
            dct_patch = torch.abs(dct_patch)

            patch_feats = []

            for (f1, f2) in bands:
                f2 = min(f2, min(patch_h, patch_w))

                band = dct_patch[:, f1:f2, f1:f2]
                flat = band.reshape(C, -1)

                mean = flat.mean(dim=1)
                var = flat.var(dim=1)
                std = torch.sqrt(var + 1e-8)
                z = (flat - mean[:, None]) / std[:, None]

                skew = torch.mean(z**3, dim=1)
                kurt = torch.mean(z**4, dim=1)

                stats = torch.stack([mean, var, skew, kurt], dim=1)
                patch_feats.append(stats)

            patch_feats = torch.cat(patch_feats, dim=1).reshape(-1)
            features.append(patch_feats)

    features = torch.cat(features, dim=0)
    features = (features - features.mean()) / (features.std() + 1e-6)

    return features  # 576-D



parser = argparse.ArgumentParser(description='DE-FAKE finetuning')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--inputpath_linear', type=str, default=None, help='path to pretrained linear model')
parser.add_argument('--inputpath_clip', type=str, default=None, help='path to pretrained CLIP model')
parser.add_argument('--outputpath_linear', type=str, default=None, help='path to save linear model - should be like filename.pt')
parser.add_argument('--outputpath_clip', type=str, default=None, help='path to save CLIP model - should be like filename.pt')

args = parser.parse_args()


class NeuralNet(nn.Module):
    def __init__(self, input_size=1088, num_classes=2, dropout_p=0.5):
        super(NeuralNet, self).__init__()
        
        # Layer 1: expand to capture interactions
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_p)
        
        # Layer 2: new added hidden layer
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(dropout_p)
        
        # Layer 3: bottleneck layer
        self.fc3 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_p)
        
        # Layer 4: smaller layer before classification
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout_p)
        
        # Output layer
        self.fc5 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 
size = 224
with torch.no_grad():
    x = torch.randn(1,3,224,224).to(device)
    emb = model.encode_image(x)
    print("CLIP embedding shape:", emb.shape)



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
        dct_feat = compute_patch_dct_stats(image)  # NEW
        return image, label, caption, dct_feat



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
        dct_feat = compute_patch_dct_stats(image)  # NEW
        return image, label, caption, dct_feat
 


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
# # DCT features: 3 bands × 4th-order × 3 channels × 4 patches = 576
# CLIP embedding: 512
# TOTAL input_size = 512 + 576 = 1088
# --------------------------------------

# New line using your extended 5-layer network
linear = NeuralNet(
    input_size=1600,   # same as before
    num_classes=2,     # output classes
    dropout_p=0.5      # optional, can adjust
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
    print(f"\n========== EPOCH {i+1} / {args.epoch} STARTED ==========") 
    loss_epoch = 0
    train_acc = []
    train_true = []
    
    test_acc = []
    test_true = []
    

    for step, (x, y, t, d) in enumerate(tqdm(train_loader)):
        x = x.cuda()
        y = y.cuda()
        d = d.cuda()

        linear.train()
        model.eval()

        texts = clip.tokenize(list(t), context_length=77, truncate=True).to(device)

        with torch.no_grad():
            imga_embedding = model.encode_image(x)          # (B,512)
            text_embedding = model.encode_text(texts)       # (B,512)

        # dropout for regularization
        imga_embedding = F.dropout(imga_embedding, p=0.2, training=linear.training)
        text_embedding = F.dropout(text_embedding, p=0.2, training=linear.training)
        dct_feat = F.dropout(d.float(), p=0.1, training=linear.training)

        # NEW: 1600-D combined embedding
        emb = torch.cat((imga_embedding, text_embedding, dct_feat), dim=1)
        # emb shape MUST be (batch, 1600)

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
        
    for step, (x, y, t, d) in enumerate(tqdm(val_loader)):
        x = x.cuda()
        y = y.cuda()
        d = d.cuda()
        model.eval()
        linear.eval()

        texts = clip.tokenize(list(t), context_length=77, truncate=True).to(device)

        with torch.no_grad():
            imga_embedding = model.encode_image(x)      # 512
            text_embedding = model.encode_text(texts)   # 512

        imga_embedding = F.dropout(imga_embedding, p=0.2, training=False)
        text_embedding = F.dropout(text_embedding, p=0.2, training=False)
        dct_feat = F.dropout(d.float(), p=0.1, training=False)

        # >>> FIXED: SAME CONCAT AS TRAINING <<<
        emb = torch.cat((imga_embedding, text_embedding, dct_feat), 1)  # 1600-D

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
    print(f"========== EPOCH {i+1} / {args.epoch} FINISHED ==========")
    # save model
    #torch.save(linear, args.outputpath_linear)
    #torch.save(model, args.outputpath_clip)
    torch.save(linear.state_dict(), args.outputpath_linear)
    torch.save(model.state_dict(), args.outputpath_clip)




