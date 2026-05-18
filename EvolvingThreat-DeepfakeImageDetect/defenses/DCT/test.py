# Original

#Modification save the result .csv

import os
import torch
import torch_dct as dct
import argparse
import random
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import pandas as pd
from tqdm import tqdm  # ← Add this import at the top of your script

DEVICE = "cuda:0"

# ----------------- Seeding -----------------
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------- Image Loader -----------------
def array_from_imgdir(imgdir, grayscale=True):
    """
    Load images from a directory as tensors, pad them to the largest size,
    and scale to [-1, 1]. Returns a stacked tensor.
    """
    paths = [os.path.join(imgdir, imgname) for imgname in os.listdir(imgdir)]

    if grayscale:
        def loader(path):
            return transforms.ToTensor()(Image.open(path).convert("L"))

    # Load images into a list
    array_list = Parallel(n_jobs=8)(delayed(loader)(path) for path in paths)

    # Find max height and width
    max_h = max(img.shape[1] for img in array_list)
    max_w = max(img.shape[2] for img in array_list)

    # Pad each image to max_h, max_w
    padded_list = []
    for img in array_list:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad format: (left, right, top, bottom)
        img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        padded_list.append(img_padded)

    # Stack into a single tensor and scale to [-1,1]
    array = torch.stack(padded_list)
    array = (array * 2.0) - 1.0

    print('final array shape', array.shape)
    return array


# ----------------- Model -----------------
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
        y = torch.tensor(self.labels[idx])
        return x, y


# ----------------- Evaluation -----------------
def evaluate_metrics(actual, preds, probs):
    precision = precision_score(actual, preds, zero_division=0)
    recall = recall_score(actual, preds, zero_division=0)
    f1 = f1_score(actual, preds, zero_division=0)
    accuracy = accuracy_score(actual, preds)
    ap = average_precision_score(actual, probs[:, 1])  # use prob for class 1
    return precision, recall, f1, accuracy, ap




def valid_epoch(model, criterion, valid_loader, args):
    model.eval()
    valid_loss = 0.0
    actuals, allpreds, allprobs = [], [], []

    with torch.no_grad():
        # ✅ tqdm progress bar for live batch streaming visualization
        for inputs, labels in tqdm(valid_loader, desc="Validating", leave=False):
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            actuals += labels.tolist()
            allpreds += preds.tolist()
            allprobs.append(probs.cpu())

    allprobs = torch.cat(allprobs, dim=0).numpy()
    precision, recall, f1, accuracy, ap = evaluate_metrics(actuals, allpreds, allprobs)

    print(f'precision {precision:.4f}  recall {recall:.4f}  f1 {f1:.4f}  accuracy {accuracy:.4f}  AP {ap:.4f}')

    return valid_loss / len(valid_loader), accuracy, precision, recall, f1, ap

#center_crop = transforms.CenterCrop((512, 512))
#resize_tf = transforms.Resize((1024, 1024))
# Remove center_crop entirely



# ----------------- Main -----------------
#def main(args):
def main(args):
    all_probs, all_preds, all_labels = [], [], []
    criterion = nn.CrossEntropyLoss()

    means = torch.load(os.path.join(args.path_to_mean_std, 'means.pt'))
    stds = torch.load(os.path.join(args.path_to_mean_std, 'stds.pt'))

    # ---------------------------
    # 🔹 Infer input_size 
    # ---------------------------
    input_size = 1024 * 1024  # Same as training: args.input_size*args.input_size
    print(f"[INFO] Using input size: {input_size}")
    # ---------------------------

    # Initialize model with correct input size
    model = LogisticRegression(input_size=input_size, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # ---------------------------
    # Streaming DCT computation from real/fake dirs
    # ---------------------------
    def process_dir(imgdir, label):
        paths = [os.path.join(imgdir, f) for f in os.listdir(imgdir)]
        for p in paths:
            img = Image.open(p).convert("L")  # p is defined here
            #print(img.size)  # width, height
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = (img * 2.0) - 1.0
            x_tf = dct.dct_2d(img, norm='ortho')
            x_tf = torch.log(torch.abs(x_tf) + 1e-12).squeeze(0).squeeze(0).reshape(1, -1)
            x_tf = (x_tf - means) / stds
            yield x_tf, torch.tensor([label])


    from itertools import chain
    dataloader = chain(process_dir(args.real_root, 0), process_dir(args.fake_root, 1))

    # ---------------------------
    # Evaluation with tqdm progress bar
    # ---------------------------
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Testing", leave=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    # ---------------------------
    # Metrics & result saving
    # ---------------------------
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1, accuracy, ap = evaluate_metrics(all_labels, all_preds, all_probs)
    print(f'precision {precision:.4f}  recall {recall:.4f}  f1 {f1:.4f}  accuracy {accuracy:.4f}  AP {ap:.4f}')

    results = {
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "accuracy": [accuracy],
        "AP": [ap]
    }

    result_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "Mj_surrogate_StyleCLIP_results")
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

    
# ----------------- Argument Parser -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root", type=str, required=True, help="Root of fake images directory")
    parser.add_argument("--real_root", type=str, required=True, help="Root of real images directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--path_to_mean_std", type=str, required=True, help="Path to mean/std tensors")
    parser.add_argument("--input_size", type=int, default=512, help="Size of input image (not used)")
    return parser.parse_args()


# ----------------- Entry Point -----------------
if __name__ == "__main__":
    seed_everything()
    main(parse_args())
















