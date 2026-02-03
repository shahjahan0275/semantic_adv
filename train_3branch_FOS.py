import os
import random
import numpy as np
import pandas as pd
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import clip

from models import clip_models_FOS as clip_models   #For Fourth order statistics

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer_FOS import Trainer
from options.train_options import TrainOptions

# === import your new pipeline modules ===
from models.denoiser import run_denoiser          # step 2
from models.noise_extractor_FOS import run_extractnoise  # step 3
# ❌ REMOVED: from models.clip_models import stack_features

# ==========================
# === FORCE CLIP CACHE TO SCRATCH
# ==========================
#SCRATCH_CLIP_DIR = "/speed-scratch/a_shahj/.cache/clip"
SCRATCH_CLIP_DIR = "/forcolab/home/ashahj/.cache/clip"
os.makedirs(SCRATCH_CLIP_DIR, exist_ok=True)

# Pre-download CLIP ViT-L/14 to scratch
try:
    print("Pre-downloading CLIP ViT-L/14 to scratch...")
    clip.load("ViT-L/14", device="cpu", download_root=SCRATCH_CLIP_DIR)
    print("CLIP ViT-L/14 pre-download complete.")
except Exception as e:
    raise RuntimeError(f"Failed to pre-download CLIP model to scratch: {e}")


# --------------------------------------------------
# Helper to create val options
# --------------------------------------------------
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt


# --------------------------------------------------
# Main Training
# --------------------------------------------------
if __name__ == '__main__':
    seed = 418
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    all_classes = [
        #"ADM","BigGAN","glide","Midjourney",
        #"stable_diffusion_v_1_4","stable_diffusion_v_1_5",
        #"VQDM","wukong",
        "StyleCLIP"
    ]

    real_folders, fake_folders = [], []
    for cls in all_classes:
        #real_path = f"/speed-scratch/a_shahj/D3/data/genimage_train/{cls}/0_real"
        #fake_path = f"/speed-scratch/a_shahj/D3/data/genimage_train/{cls}/1_fake"
        real_path = f"/forcolab/home/ashahj/D3/data/genimage_train/{cls}/0_real"
        fake_path = f"/forcolab/home/ashahj/D3/data/genimage_train/{cls}/1_fake"

        if opt.train_samples > 0:
            real_files = [os.path.join(real_path, f) for f in os.listdir(real_path)]
            fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path)]
            real_sample = random.sample(real_files, min(opt.train_samples, len(real_files)))
            fake_sample = random.sample(fake_files, min(opt.train_samples, len(fake_files)))
            real_folders.extend(real_sample)
            fake_folders.extend(fake_sample)
        else:
            real_folders.append(real_path)
            fake_folders.append(fake_path)

    data_loader = create_dataloader(opt, real_folders, fake_folders)

    # === VAL DATA ===
    val_loader_list = []
    val_data_root = all_classes
    for cls in val_data_root:
        #real_folders = [f"/speed-scratch/a_shahj/D3/data/genimage_val/{cls}/0_real"]
        #fake_folders = [f"/speed-scratch/a_shahj/D3/data/genimage_val/{cls}/1_fake"]
        real_folders = [f"/forcolab/home/ashahj/D3/data/genimage_val/{cls}/0_real"]
        fake_folders = [f"/forcolab/home/ashahj/D3/data/genimage_val/{cls}/1_fake"]
        val_loader_list.append(create_dataloader(val_opt, real_folders, fake_folders))

    # initialize detector
    model = Trainer(opt)

    # writers
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    # early stopping
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=0.0, verbose=True)

    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))
    results_dict = {}

    # === MAIN TRAIN LOOP ===
    for epoch in range(opt.niter):
        for i, data in enumerate(tqdm(data_loader)):
            model.total_steps += 1

            # === NEW PIPELINE ===
            # Step 1: original image tensor
            #original = data["image"].to("cuda")  # [B,C,H,W]
            original = data[0].to("cuda")  # [B,C,H,W]

            # Step 2: denoise
            denoised = run_denoiser(original)

            # Step 3: extract noise features (adapt output_dim to CLIP backbone)
            #clip_out_dim = model.clip_encoder.output_dim  # assume Trainer exposes this
            # Step 3: extract noise features (adapt output_dim to CLIP backbone)
            # Trainer does not expose clip_encoder — derive the CLIP backbone name from opt.arch
            try:
                arch_name = opt.arch.split(':', 1)[1] if ':' in opt.arch else opt.arch
            except Exception:
                arch_name = opt.arch

            # prefer the penultimate key if present, else fallback to backbone key
            penultimate_key = arch_name + "-penultimate"
            if hasattr(clip_models, "CHANNELS") and penultimate_key in clip_models.CHANNELS:
                clip_out_dim = clip_models.CHANNELS[penultimate_key]
            elif hasattr(clip_models, "CHANNELS") and arch_name in clip_models.CHANNELS:
                clip_out_dim = clip_models.CHANNELS[arch_name]
            else:
                # safe fallback: use ViT-L/14 penultimate dims (most common in your setup)
                clip_out_dim = clip_models.CHANNELS.get("ViT-L/14-penultimate", 1024)


            noise_features = run_extractnoise(original, denoised, output_dim=clip_out_dim)

            # === NO MORE stack_features ===
            # The CLIP model now handles (original, denoised, noise) inside forward()
            model.set_input(data, extra_features=(original, denoised, noise_features))
            model.optimize_parameters()

            # === Logging ===
            if model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {model.loss} at step: {model.total_steps}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time() - start_time) / model.total_steps))

        # === SAVE MODELS ===
        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save_networks('model_epoch_best.pth')
            model.save_networks(f'model_epoch_{epoch}.pth')

        # === VALIDATION ===
        model.eval()
        acc_list, ap_list, b_acc_list, threshold_list = [], [], [], []
        y_pred_list, y_true_list = [], []
        for i, val_loader in enumerate(val_loader_list):
            ap, r_acc0, f_acc0, acc, r_acc1, f_acc1, acc1, best_thres, y_pred, y_true = validate(
                model.model, val_loader, find_thres=True
            )
            acc_list.append(acc)
            ap_list.append(ap)
            b_acc_list.append(acc1)
            threshold_list.append(best_thres)

            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)

            print(f"(Val on {val_data_root[i]} @ epoch {epoch}) "
                  f"acc: {acc}; ap: {ap}; r_acc0:{r_acc0}, f_acc0:{f_acc0}, "
                  f"r_acc1:{r_acc1}, f_acc1:{f_acc1}, acc1:{acc1}, best_thres:{best_thres}")
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)

        # === AVERAGE METRICS ===
        avg_ap = sum(ap_list) / len(val_loader_list)
        avg_acc = sum(acc_list) / len(val_loader_list)
        avg_b_acc = sum(b_acc_list) / len(val_loader_list)
        avg_threshold = sum(threshold_list) / len(val_data_root)

        results_dict[f'epoch_{epoch}_ap'] = ap_list + [avg_ap]
        results_dict[f'epoch_{epoch}_acc'] = acc_list + [avg_acc]
        results_dict[f'epoch_{epoch}_b_acc'] = b_acc_list + [avg_b_acc]
        results_dict[f'epoch_{epoch}_b_threshold'] = threshold_list + [avg_threshold]

        results_df = pd.DataFrame(results_dict)
        results_df.to_excel(
            os.path.join(opt.checkpoints_dir, opt.name, 'results.xlsx'),
            sheet_name='sheet1',
            index=False
        )
        print(f"(average Val on all dataset @ epoch {epoch}) acc: {avg_acc}; ap: {avg_ap}")

        np.savez(os.path.join(opt.checkpoints_dir, opt.name, f'y_pred_eval_{epoch}.npz'), *y_pred_list)
        np.savez(os.path.join(opt.checkpoints_dir, opt.name, f'y_true_eval_{epoch}.npz'), *y_true_list)

        # === EARLY STOPPING ===
        early_stopping(avg_acc)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=0.0, verbose=True)
            else:
                print("Early stopping. Training finished.")
                break

        model.train()
