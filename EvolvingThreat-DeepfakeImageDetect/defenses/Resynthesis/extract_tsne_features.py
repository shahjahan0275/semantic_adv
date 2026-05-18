import torch
import numpy as np
from tqdm import tqdm


def extract_features(
    loader,
    model,
    sr_model,
    perception_net,
    args,
    gpu,
    dct_extractor=None
):
    model.eval()
    sr_model.eval()
    if dct_extractor is not None:
        dct_extractor.eval()

    feats, labels = [], []

    with torch.no_grad():
        for input, target, _ in loader:

            input = input.cuda(gpu, non_blocking=True)
            target = target.numpy()

            # ---- SAME preprocessing as test() ----
            lr = 0
            for ii in range(args.sr_scale):
                for jj in range(args.sr_scale):
                    lr += input[:, :, ii::args.sr_scale, jj::args.sr_scale] \
                          / (args.sr_scale * args.sr_scale)

            lr = lr / 255.0
            input = input / 255.0

            preds_input = sr_model(lr)

            if args.idx_stages > 0:
                per_rec = perception_net(preds_input)
                per_gt  = perception_net(input)
                rec_features = torch.abs(
                    per_rec[args.idx_stages - 1] -
                    per_gt[args.idx_stages - 1]
                )
            else:
                rec_features = torch.abs(preds_input - input)

            # ---- DCT branch (ONLY if provided) ----
            if dct_extractor is not None:
                dct_stats = dct_extractor(rec_features)
                _, feat = model(rec_features, dct_stats)
            else:
                _, feat = model(rec_features)

            feats.append(feat.cpu().numpy())
            labels.append(target)

    return np.concatenate(feats), np.concatenate(labels)

