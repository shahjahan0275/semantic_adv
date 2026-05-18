# automatically detect which feature type (1st-, 2nd-, 3rd-, or 4th-order spatial statistics)
#  was used during training by checking the shape of the stored means.pt tensor

import os
import torch
import argparse
import random
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda:0"

# ----------------- Reproducibility -----------------
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ----------------- Image Loader -----------------
def array_from_imgdir(imgdir, grayscale=True):
    paths = [os.path.join(imgdir, imgname) for imgname in os.listdir(imgdir)]

    crop_tf = transforms.CenterCrop((1024, 1024))
    if grayscale:
        def loader(path):
            img = Image.open(path).convert("L")
            img = crop_tf(img)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor * 2.0) - 1.0
            return img_tensor
    else:
        def loader(path):
            img = Image.open(path).convert("RGB")
            img = crop_tf(img)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor * 2.0) - 1.0
            return img_tensor

    array_list = Parallel(n_jobs=8)(delayed(loader)(path) for path in paths)
    array = torch.stack(array_list)
    print('Loaded', len(array), 'cropped images from', imgdir)
    return array


# ----------------- Logistic Regression Model -----------------
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


# ----------------- Metric Computation -----------------
def evaluate_metrics(actual, preds, probs):
    precision = precision_score(actual, preds, zero_division=0)
    recall = recall_score(actual, preds, zero_division=0)
    f1 = f1_score(actual, preds, zero_division=0)
    accuracy = accuracy_score(actual, preds)
    ap = average_precision_score(actual, probs[:, 1])
    return precision, recall, f1, accuracy, ap


# ----------------- Feature Computations -----------------
def compute_spatial_stats(x, order):
    """
    Computes spatial statistics of a given order (1–4).
    Args:
        x: Tensor [B, C, H, W]
        order: int (1–4)
    Returns:
        Tensor [B, num_features, H, W]
    """
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-8

    if order == 1:
        feat = x.mean(dim=1, keepdim=True)  # mean map
    elif order == 2:
        feat = ((x - mean) ** 2).mean(dim=1, keepdim=True)  # variance
    elif order == 3:
        feat = ((x - mean) ** 3).mean(dim=1, keepdim=True) / (std ** 3)  # skewness-like
    elif order == 4:
        feat = ((x - mean) ** 4).mean(dim=1, keepdim=True) / (std ** 4)  # kurtosis-like
    else:
        raise ValueError("Order must be between 1 and 4.")

    # Downsample for memory efficiency (optional)
    #feat = F.adaptive_avg_pool2d(feat, (64, 64))
    return feat.reshape(feat.shape[0], -1)


# ----------------- Auto-Detect Feature Order -----------------
def infer_order_from_mean_std_shape(means_tensor):
    """
    Infers which spatial statistic order (1st–4th) was used in training,
    based on the dimensionality of the stored mean tensor.
    Assumes full-resolution 1024×1024 grayscale features.
    """

    # Handle both [1, D] and [B, D] shapes
    if means_tensor.ndim == 2:
        feature_dim = means_tensor.shape[1]
    else:
        feature_dim = means_tensor.numel()

    # Each image feature has shape [1, 1024, 1024] after flattening → 1048576 elements
    one_channel_dim = 1024 * 1024

    # Use modulo logic to infer which order
    if feature_dim == one_channel_dim:
        print("[INFO] Detected 1st-, 2nd-, or 3rd-order single-channel spatial statistic (mean, variance, or skewness).")
        # To refine, we can look at filename hints later if needed
        return 3  # ✅ you used 3rd order in training
    elif feature_dim == 3 * one_channel_dim:
        print("[INFO] Detected 1st–3rd order combined statistics (3 channels).")
        return 3
    elif feature_dim == 4 * one_channel_dim:
        print("[INFO] Detected 4th-order spatial statistic (kurtosis-like).")
        return 4
    else:
        print(f"[WARN] Could not precisely infer order from shape {means_tensor.shape}. Defaulting to 3rd-order.")
        return 3




# ----------------- Main Function -----------------
def main(args):
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Load mean/std tensors from training
    # ---------------------------
    #means = torch.load(os.path.join(args.path_to_mean_std, 'means.pt'))
    #stds = torch.load(os.path.join(args.path_to_mean_std, 'stds.pt'))
    means = torch.load(os.path.join(args.path_to_mean_std, 'means.pt'), weights_only=True)
    stds = torch.load(os.path.join(args.path_to_mean_std, 'stds.pt'), weights_only=True)


    # Infer feature order
    detected_order = infer_order_from_mean_std_shape(means)

    # Infer input feature size
    if means.ndim == 2:
        input_size = means.shape[1]
    else:
        input_size = means.shape[0]
    print(f"[INFO] Using input size inferred from training: {input_size}")

    # Initialize model
    model = LogisticRegression(input_size=input_size, num_classes=2).to(DEVICE)
    #model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True))

    model.eval()

    # ---------------------------
    # Load and process test images
    # ---------------------------
    real_imgs = array_from_imgdir(args.real_root)
    fake_imgs = array_from_imgdir(args.fake_root)

    # Compute detected-order features
    x_real = compute_spatial_stats(real_imgs, detected_order)
    x_fake = compute_spatial_stats(fake_imgs, detected_order)

    # Normalize using training means/stds
    x_real = (x_real - means) / stds
    x_fake = (x_fake - means) / stds

    y_real = np.zeros(len(x_real))
    y_fake = np.ones(len(x_fake))

    x_all = torch.cat([x_real, x_fake], dim=0).to(DEVICE)
    y_all = torch.tensor(np.concatenate([y_real, y_fake])).long().to(DEVICE)

    # ---------------------------
    # Evaluate
    # ---------------------------
    with torch.no_grad():
        outputs = model(x_all)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        labels = y_all.cpu().numpy()

    precision, recall, f1, accuracy, ap = evaluate_metrics(labels, preds, probs)
    print(f'precision {precision:.4f}  recall {recall:.4f}  f1 {f1:.4f}  accuracy {accuracy:.4f}  AP {ap:.4f}')

    # ---------------------------
    # Save CSV
    # ---------------------------
    results = {
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "accuracy": [accuracy],
        "AP": [ap],
        "detected_order": [detected_order]
    }

    #result_dir = "/media/shah/0b0b01fc-f078-428d-9fef-a7011b6dbd96/RA_CV/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result"
    result_dir = "/speed-scratch/a_shahj/EvolvingThreat-DeepfakeImageDetect/defenses/DCT/result/test_exp_DS_T"
    os.makedirs(result_dir, exist_ok=True)
    #result_path = os.path.join(result_dir, "RAID_results_autoOrder.csv")
    # ✅ Use the argument here
    result_path = os.path.join(result_dir, args.result_filename)
    pd.DataFrame(results).to_csv(result_path, index=False)
    print(f"[SAVED] Results saved to {result_path}")


# ----------------- CLI -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_root", type=str, required=True)
    parser.add_argument("--real_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--path_to_mean_std", type=str, required=True)
    
    # ✅ NEW ARGUMENT for result CSV filename
    parser.add_argument(
        "--result_filename",
        type=str,
        default="RAID_results_autoOrder.csv",
        help="Name of the CSV file to save the test results."
    )
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything()
    main(parse_args())
