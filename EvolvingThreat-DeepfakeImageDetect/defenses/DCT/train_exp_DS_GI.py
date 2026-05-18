# Fixed resolution Data from multiple training and  validation generator
import os 
import torch
import torch_dct as dct
import argparse
import random 
import numpy as np
import torchvision.transforms as transforms 
from pathlib import Path
from PIL import Image 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count




DEVICE = "cuda:0"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def array_from_imgdir(imgdir, grayscale=True, max_samples=None, seed=42, input_size=1024):
    """
    Two-pass optimized image loader (cluster-safe, SLURM-aware) with tqdm progress bars.
    """

    # === Determine safe number of cores for this SLURM job ===
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")
    if slurm_cpus and slurm_cpus.isdigit():
        n_cores = int(slurm_cpus)
    else:
        n_cores = max(1, cpu_count() - 1)

    target_size = (input_size, input_size)
    random.seed(seed)

    print(f"[1/2] Scanning headers in {imgdir} using {n_cores} CPU cores...")

    all_paths = [os.path.join(imgdir, f) for f in os.listdir(imgdir)]
    all_paths = [p for p in all_paths if os.path.isfile(p)]

    # ---- Helper for Pass 1 ----
    def check_header(path):
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w >= target_size[0] and h >= target_size[1]:
                    return path
        except (UnidentifiedImageError, OSError):
            return None
        return None

    # ---- Parallel header scan with progress ----
    valid_paths = []
    with tqdm(total=len(all_paths), desc="Header Scan", ncols=100, unit="img", colour="cyan") as pbar:
        def wrapper(path):
            res = check_header(path)
            pbar.update(1)
            return res

        valid_paths = Parallel(n_jobs=n_cores, prefer="threads")(
            delayed(wrapper)(p) for p in all_paths
        )

    valid_paths = [p for p in valid_paths if p is not None]

    print(f"  Found {len(valid_paths)} valid images ≥ {target_size[0]}x{target_size[1]} out of {len(all_paths)} total.")

    if len(valid_paths) == 0:
        print(f"[ERROR] No valid images found in {imgdir}")
        return torch.empty(0)

    # ---- Random sampling ----
    if max_samples is not None and len(valid_paths) > max_samples:
        random.seed(seed)
        valid_paths = random.sample(valid_paths, max_samples)
        print(f"  Randomly selected {len(valid_paths)} samples for loading.")

    # ---- Pass 2: Load selected images ----
    crop_tf = transforms.CenterCrop(target_size)

    def loader(path):
        try:
            img = Image.open(path)
            if grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

            img = crop_tf(img)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor * 2.0) - 1.0  # scale to [-1, 1]
            return img_tensor
        except (UnidentifiedImageError, OSError):
            return None

    print(f"[2/2] Loading {len(valid_paths)} selected images into memory...")

    load_threads = min(4, n_cores)
    results = []
    with tqdm(total=len(valid_paths), desc="Image Load", ncols=100, unit="img", colour="green") as pbar:
        def load_wrap(path):
            res = loader(path)
            pbar.update(1)
            return res

        results = Parallel(n_jobs=load_threads, prefer="threads")(
            delayed(load_wrap)(p) for p in valid_paths
        )

    valid_imgs = [r for r in results if r is not None]

    if len(valid_imgs) == 0:
        print(f"[ERROR] No images successfully loaded from {imgdir}")
        return torch.empty(0)

    array = torch.stack(valid_imgs)
    print(f"✅ Loaded {len(array)} valid {target_size} images from {imgdir}")
    return array



def load_balanced_data(base_dir, num_real, num_fake, seed=42, grayscale=True, input_size=1024):

    """
    Loads equal number of real and fake images from each generator subfolder.
    Skips a generator entirely if its images fail the target resolution check.
    """
    generators = [ "ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5","VQDM", "wukong", "StyleCLIP_dataset"]
    #generators = ["Midjourney", "StyleCLIP_dataset"]
    random.seed(seed)

    def adjust_count(n):
        return ((n + len(generators) - 1) // len(generators)) * len(generators)

    num_real = adjust_count(num_real)
    num_fake = adjust_count(num_fake)
    per_class_real = num_real // len(generators)
    per_class_fake = num_fake // len(generators)

    print(f"\n[INFO] Adjusted counts: real={num_real}, fake={num_fake}")
    print(f"[INFO] => {per_class_real} real + {per_class_fake} fake per generator.\n")

    all_real, all_fake = [], []

    for gen in generators:
        real_dir = Path(base_dir) / gen / "0_real"
        fake_dir = Path(base_dir) / gen / "1_fake"
        #real_dir = Path(base_dir) / gen / "real"
        #fake_dir = Path(base_dir) / gen / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            print(f"[WARN] Missing generator folder: {gen}, skipping.")
            continue

        #real_imgs = array_from_imgdir(real_dir, grayscale=grayscale, max_samples=per_class_real, seed=seed)
        #fake_imgs = array_from_imgdir(fake_dir, grayscale=grayscale, max_samples=per_class_fake, seed=seed + 1)
        real_imgs = array_from_imgdir(real_dir, grayscale=grayscale, max_samples=per_class_real, seed=seed, input_size=input_size)
        fake_imgs = array_from_imgdir(fake_dir, grayscale=grayscale, max_samples=per_class_fake, seed=seed + 1, input_size=input_size)



        # ✅ Skip entire generator if it doesn't have valid images
        if real_imgs.numel() == 0 or fake_imgs.numel() == 0:
            print(f"[SKIP CLASS] {gen}: skipped due to invalid or mismatched resolution images.\n")
            continue

        print(f"[GENERATOR] {gen}: Loaded {len(real_imgs)} real, {len(fake_imgs)} fake images.")
        all_real.append(real_imgs)
        all_fake.append(fake_imgs)

    # Final safety check
    if len(all_real) == 0 or len(all_fake) == 0:
        raise RuntimeError("[FATAL] No valid generator class passed the resolution check.")

    all_real = torch.cat(all_real, dim=0)
    all_fake = torch.cat(all_fake, dim=0)

    print(f"\n[SUMMARY] Total real images: {len(all_real)}, Total fake images: {len(all_fake)}\n")
    return all_real, all_fake





# define a logistic regression model
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

def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    with tqdm(total=len(train_loader), desc="Training", ncols=100, unit="batch", colour="yellow") as pbar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_acc += (preds == labels).float().mean().item()

            pbar.update(1)
            pbar.set_postfix(loss=f"{train_loss / pbar.n:.4f}", acc=f"{train_acc / pbar.n:.4f}")

    return train_loss / len(train_loader), train_acc / len(train_loader)


def valid_epoch(model, criterion, valid_loader):
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0

    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Validation", ncols=100, unit="batch", colour="magenta") as pbar:
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                valid_acc += (preds == labels).float().mean().item()

                pbar.update(1)
                pbar.set_postfix(loss=f"{valid_loss / pbar.n:.4f}", acc=f"{valid_acc / pbar.n:.4f}")

    return valid_loss / len(valid_loader), valid_acc / len(valid_loader)







def main(args):
    
    # ===================== 🔹 STEP 1: Load balanced data per generator =====================
    train_root = args.train_root
    val_root = args.val_root

    real_train, fake_train = load_balanced_data(
        base_dir=train_root,
        num_real=args.num_real_train,
        num_fake=args.num_fake_train,
        seed=42,
        input_size=args.input_size
    )
    print(f"Selected {len(real_train)} real and {len(fake_train)} fake samples for TRAINING.")

    real_val, fake_val = load_balanced_data(
        base_dir=val_root,
        num_real=args.num_real_val,
        num_fake=args.num_fake_val,
        seed=99,
        input_size=args.input_size
    )
    print(f"Selected {len(real_val)} real and {len(fake_val)} fake samples for VALIDATION.")


    # ======== 🔹 Combine validation data ========
    x_train = torch.cat([real_train, fake_train], dim=0)
    y_train = torch.tensor([0.0] * len(real_train) + [1.0] * len(fake_train))
    del real_train, fake_train

    x_val = torch.cat([real_val, fake_val], dim=0)
    y_val = torch.tensor([0.0] * len(real_val) + [1.0] * len(fake_val))
    del real_val, fake_val

    # ===================== 🔹 STEP 5: Feature extraction =====================
    print('feature calculation...')
    x_train_tf = dct.dct_2d(x_train, norm='ortho')
    x_train_tf = torch.log(torch.abs(x_train_tf) + 1e-12)

    x_val_tf = dct.dct_2d(x_val, norm='ortho')
    x_val_tf = torch.log(torch.abs(x_val_tf) + 1e-12)

    x_train_tf = x_train_tf.squeeze(1)
    x_val_tf = x_val_tf.squeeze(1)

    x_train_tf = x_train_tf.reshape(x_train_tf.shape[0], -1)
    x_val_tf = x_val_tf.reshape(x_val_tf.shape[0], -1)
    print('reshaped...')

    # ===================== 🔹 STEP 6: Normalization =====================
    means = x_train_tf.mean(0, keepdim=True)
    stds = x_train_tf.std(0, unbiased=False, keepdim=True)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(means, os.path.join(args.save_dir, 'means.pt'))
    torch.save(stds, os.path.join(args.save_dir, 'stds.pt'))

    x_train_tf = (x_train_tf - means) / stds
    x_val_tf = (x_val_tf - means) / stds
    print("rescaled...")

    # ===================== 🔹 STEP 7: Dataset & DataLoader =====================
    train_dataset = MyDataset(x_train_tf, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = MyDataset(x_val_tf, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # ===================== 🔹 STEP 8: Model setup =====================
    print('training model...')
    input_size = args.input_size * args.input_size
    model = LogisticRegression(input_size, num_classes=2)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.epochs
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        valid_loss, valid_acc = valid_epoch(model, criterion, val_loader)

        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}')

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_MjStyle.pth'))


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument(
        #"image_root",
        #type=Path,
        #help="Root of image directory containing 'train', 'val', and test.",
    #)
    parser.add_argument(
        "--train_root",
        type=Path,
        default=Path("/speed-scratch/a_shahj/D3/data/genimage_train/"),
        help="Root folder for training data (contains 8 generator subfolders)."
    )
    parser.add_argument(
        "--val_root",
        type=Path,
        default=Path("/media/shah/SSD990PRO/RA_CV/D3/data/genimage_val/"),
        help="Root folder for validation data (contains 8 generator subfolders)."
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Size of input image",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay regularization value",
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="checkpoints_rgb")
    
    parser.add_argument(
        "--num_real_train", 
        type=int, 
        default=400)
    
    parser.add_argument(
        "--num_fake_train", 
        type=int, 
        default=400)
    parser.add_argument(
        "--num_real_val", 
        type=int, 
        default=50)
    parser.add_argument(
        "--num_fake_val", 
        type=int, 
        default=50)

    return parser.parse_args()

if __name__ == "__main__":
    seed_everything()

    main(parse_args())
