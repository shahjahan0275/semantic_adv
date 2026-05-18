from options.test_options import TestOptions
from models import create_model
import numpy as np
import os
import torch
from utils import pidfile, util, imutil, pbar
from torch.utils.data import DataLoader
from data.unpaired_dataset import UnpairedDataset
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cudnn.benchmark = True


def run_eval(opt, output_dir):
    device = torch.device(
        f"cuda:{opt.gpu_ids[0]}" if len(opt.gpu_ids) > 0 else "cpu"
    )

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    fake_label = opt.fake_class_id
    real_label = 1 - fake_label

    paths = []
    prediction_voted = []
    prediction_avg_after_softmax = []
    prediction_avg_before_softmax = []
    prediction_raw = []
    labels = []

    with torch.no_grad():
        for data_path, label in zip(
            [opt.real_im_path, opt.fake_im_path],
            [real_label, fake_label]
        ):
            dset = UnpairedDataset(opt, data_path, is_val=True)
            dl = DataLoader(
                dset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.nThreads,
                pin_memory=True
            )

            for data in dl:
                ims = data['img'].to(device)
                pred_labels = torch.full(
                    (ims.size(0),),
                    label,
                    dtype=torch.long,
                    device=device
                )

                inputs = dict(ims=ims, labels=pred_labels)

                model.reset()
                model.set_input(inputs)
                model.test(True)

                predictions = model.get_predictions()

                labels.append(pred_labels.cpu().numpy())
                prediction_voted.append(predictions.vote)
                prediction_avg_before_softmax.append(predictions.before_softmax)
                prediction_avg_after_softmax.append(predictions.after_softmax)
                prediction_raw.append(predictions.raw)
                paths.extend(data['path'])

    # ---- Metrics ----
    if opt.model == 'patch_discriminator':
        compute_metrics(
            np.concatenate(prediction_voted),
            np.concatenate(labels),
            os.path.join(output_dir, 'metrics_voted')
        )

        compute_metrics(
            np.concatenate(prediction_avg_before_softmax),
            np.concatenate(labels),
            os.path.join(output_dir, 'metrics_avg_before_softmax')
        )

        compute_metrics(
            np.concatenate(prediction_avg_after_softmax),
            np.concatenate(labels),
            os.path.join(output_dir, 'metrics_avg_after_softmax')
        )

        patch_preds = np.concatenate(prediction_raw, axis=0)
        patch_preds = patch_preds.transpose(0, 2, 3, 1)
        n, h, w, c = patch_preds.shape

        patch_labels = np.concatenate(labels)[:, None, None]
        patch_labels = np.tile(patch_labels, (1, h, w))

        patch_preds = patch_preds.reshape(-1, 2)
        patch_labels = patch_labels.reshape(-1)

        compute_metrics(
            patch_preds,
            patch_labels,
            os.path.join(output_dir, 'metrics_patch'),
            #plot=False
        )
    else:
        compute_metrics(
            np.concatenate(prediction_raw),
            np.concatenate(labels),
            os.path.join(output_dir, 'metrics')
        )


def compute_metrics(predictions, labels, save_path):
    print(f"Computing metrics for {save_path}")

    tp = fp = tn = fn = 0
    for i in range(len(labels)):
        pred = np.argmax(predictions[i])
        if labels[i] == 0:
            tp += int(pred == 0)
            fn += int(pred != 0)
        else:
            fp += int(pred == 0)
            tn += int(pred != 0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    print(
        f"precision={precision:.4f}, "
        f"recall={recall:.4f}, "
        f"f1={f1:.4f}, "
        f"accuracy={accuracy:.4f}"
    )


if __name__ == '__main__':
    opt = TestOptions().parse()

    print(f"Evaluating model: {opt.name} epoch {opt.which_epoch}")
    print(f"Real images: {opt.real_im_path}")
    print(f"Fake images: {opt.fake_im_path}")

    output_dir = os.path.join(
        opt.results_dir,
        opt.name,
        opt.partition,
        f'epoch_{opt.which_epoch}',
        opt.dataset_name
    )

    os.makedirs(output_dir, exist_ok=True)
    run_eval(opt, output_dir)
