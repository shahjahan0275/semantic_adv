import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from tqdm import tqdm
from Pda_Detector import Detector, default_image_loader, activation_prune


# -------------------------------------------------------------------------
# PDA inference helper (replacement for missing pda_infer_one)
# -------------------------------------------------------------------------
def pda_infer_one(image_path, detector, reference_Z, tau, device,
                  k=5, size=224, prune_p=90.0, reduction="pca",
                  reduction_model=None):
    """
    Perform PDA inference on a single image.

    Args:
        image_path (str): path to test image
        detector (Detector): trained model
        reference_Z (np.ndarray): reduced reference features (e.g., PCA)
        tau (float): threshold from calibration
        device (torch.device)
        reduction_model: fitted PCA or TSNE object
    Returns:
        pred_label (str): predicted class label
        score (float): mean distance to k nearest reference samples
    """
    # 1️⃣ Load and extract features
    x = default_image_loader(image_path, size=size).unsqueeze(0).to(device)
    with torch.no_grad():
        f = detector.extract_features(x).cpu().numpy()

    # 2️⃣ Activation pruning
    f_pruned = activation_prune(f, percentile=prune_p)

    # 3️⃣ Dimensionality reduction
    if reduction_model is not None:
        Z_test = reduction_model.transform(f_pruned)
    else:
        Z_test = f_pruned

    # 4️⃣ k-NN distance computation
    nbrs = NearestNeighbors(n_neighbors=k).fit(reference_Z)
    distances, _ = nbrs.kneighbors(Z_test)
    dk = np.mean(distances)

    # 5️⃣ Decision
    pred_label = "real" if dk < tau else "fake_known"
    return pred_label, float(dk)


# -------------------------------------------------------------------------
# Utility: Load detector
# -------------------------------------------------------------------------
def load_detector(detector_path, device):
    detector = Detector(backbone_name="resnet50", pretrained=False).to(device)
    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.eval()
    print(f"✅ Loaded detector from {detector_path}")
    return detector


# -------------------------------------------------------------------------
# Utility: Load reference + calibration
# -------------------------------------------------------------------------
def load_reference_and_calibration(ref_npy, calib_npy, pca_path):
    ref_data = np.load(ref_npy, allow_pickle=True)
    calib_data = np.load(calib_npy, allow_pickle=True)
    ref_pruned = ref_data["pruned"]
    tau = float(calib_data["tau"])

    # Load the same PCA model used during training
    pca_model = joblib.load(pca_path)
    ref_pca = pca_model.transform(ref_pruned)
    print(f"✅ Loaded reference ({ref_pca.shape[0]}) and tau={tau:.6f}")
    return ref_pca, tau, pca_model


# -------------------------------------------------------------------------
# Test a folder (real or fake)
# -------------------------------------------------------------------------
def test_folder(folder_path, label, detector, ref_pca, tau, device, pca_model,
                prune_p=90.0, k_nn=5, size=224):
    files = sorted([str(p) for p in Path(folder_path).rglob("*")
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    results = []
    for f in tqdm(files, desc=f"Testing {label}", ncols=90):
        try:
            pred_label, score = pda_infer_one(
                image_path=f,
                detector=detector,
                reference_Z=ref_pca,
                tau=tau,
                device=device,
                k=k_nn,
                size=size,
                prune_p=prune_p,
                reduction="pca",
                reduction_model=pca_model,
            )
            results.append((f, label, pred_label, score))
        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")
            continue
    return results


# -------------------------------------------------------------------------
# Compute accuracy + confusion matrix + AP
# -------------------------------------------------------------------------
def compute_metrics(results):
    y_true = [r[1] for r in results]
    y_pred = [r[2] for r in results]
    y_scores = [r[3] for r in results]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake_known"])
    y_true_bin = [0 if t == "real" else 1 for t in y_true]
    ap = average_precision_score(y_true_bin, y_scores)

    print("\n================ PDA TEST SUMMARY ================")
    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\n✅ Average Precision (AP): {ap:.4f}")
    print("=================================================\n")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test PDA Detector on Real and Fake datasets")

    parser.add_argument("--detector-path", type=str, required=True)
    parser.add_argument("--ref-npz", type=str, required=True)
    parser.add_argument("--calib-npz", type=str, required=True)
    parser.add_argument("--pca-model", type=str, required=True)
    parser.add_argument("--test-real", type=str, required=True)
    parser.add_argument("--test-fake", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--prune-p", type=float, default=90.0)
    parser.add_argument("--k-nn", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load everything
    detector = load_detector(args.detector_path, device)
    ref_pca, tau, pca_model = load_reference_and_calibration(args.ref_npz, args.calib_npz, args.pca_model)

    # Run tests
    real_results = test_folder(args.test_real, "real", detector, ref_pca, tau, device, pca_model,
                               prune_p=args.prune_p, k_nn=args.k_nn, size=args.size)
    fake_results = test_folder(args.test_fake, "fake_known", detector, ref_pca, tau, device, pca_model,
                               prune_p=args.prune_p, k_nn=args.k_nn, size=args.size)

    # Compute summary metrics
    all_results = real_results + fake_results
    compute_metrics(all_results)


if __name__ == "__main__":
    main()
