import os
import sys
import time
import torch
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer_dct import Trainer
from options.train_options import TrainOptions


""" Build validation options """
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ["pil"]

    # Fix blur and JPEG ranges
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"
    val_opt = get_val_opt()


    # --------------------------------------------------------------
    # 1. Create dataloader
    # --------------------------------------------------------------
    data_loader = create_dataloader(opt)
    train_dataset = data_loader.dataset
    print("# training images =", len(train_dataset))
    print("# training batches =", len(data_loader))

    # --------------------------------------------------------------
    # 2. Compute GLOBAL DCT mean/std BEFORE training
    # --------------------------------------------------------------
    print("\n=> Computing DCT mean/std BEFORE training begins ...")

    # pre-iterate
    for _ in data_loader:
        pass

    # If ConcatDataset (real + fake)
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        for ds in train_dataset.datasets:
            if hasattr(ds, "finalize_stats"):
                ds.finalize_stats()
    else:
        train_dataset.finalize_stats()


    print("=> DCT mean/std computed successfully!\n")


    # --------------------------------------------------------------
    # 3. Create Trainer and attach dataset
    # --------------------------------------------------------------
    model = Trainer(opt)
    model.train_dataset = train_dataset   # must be attached BEFORE first save()

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )

    # --------------------------------------------------------------
    # 4. TRAINING LOOP
    # --------------------------------------------------------------
    for epoch in range(opt.niter):
        model.epoch = epoch
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {model.loss} at step: {model.total_steps}")
                train_writer.add_scalar("loss", model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print(f"Saving latest model (epoch {epoch}, step {model.total_steps})")
                model.save_networks("latest")

        # ----------------------------------------------------------
        # Save epoch model
        # ----------------------------------------------------------
        print(f"Saving epoch {epoch}...")
        model.save_networks("latest")
        model.save_networks(epoch)

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        model.eval()
        acc, ap = validate(model, val_opt)[:2]
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc}, ap: {ap}")

        # Save checkpoint explicitly with epoch ID
        model.save_networks(f"epoch_{epoch}")

        # ----------------------------------------------------------
        # Early Stopping Logic
        # ----------------------------------------------------------
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate reduced. Continuing training with new patience.")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                )
            else:
                print("Early stopping triggered.")
                break

        model.train()

