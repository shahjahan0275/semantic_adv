import numpy as np
import matplotlib
matplotlib.use("Agg")   # 🔑 IMPORTANT for SLURM
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ---- Load features ----
X1 = np.load("tsne_orig.npy")   # (N, 2048)
X2 = np.load("tsne_dct.npy")    # (N, 4096)
y  = np.load("labels.npy")

y = np.concatenate([y, y])      # (2N,)

# ---- Reduce BOTH to same dimension ----
pca_dim = 256

pca1 = PCA(n_components=pca_dim, random_state=42)
pca2 = PCA(n_components=pca_dim, random_state=42)

X1_pca = pca1.fit_transform(X1)
X2_pca = pca2.fit_transform(X2)

# ---- Stack safely ----
X = np.vstack([X1_pca, X2_pca])
domain = np.array([0]*len(X1_pca) + [1]*len(X2_pca))

# ---- t-SNE ----
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=2000,
    random_state=42,
    init="pca"
)
Z = tsne.fit_transform(X)

print("Z:", Z.shape)
print("domain:", domain.shape)
print("y:", y.shape)


# ---- Plot ----
'''
plt.figure(figsize=(8,6))

plt.scatter(
    Z[domain==0, 0], Z[domain==0, 1],
    c=y, cmap="cool", alpha=0.5, label="Original"
)

plt.scatter(
    Z[domain==1, 0], Z[domain==1, 1],
    c=y, cmap="autumn", alpha=0.5, label="DCT"
)

plt.legend()
plt.title("t-SNE of Resynthesis Features (PCA → t-SNE)")
plt.tight_layout()

# 🔑 SAVE instead of show
plt.savefig("tsne_resynthesis_pca_tsne.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved t-SNE plot to tsne_resynthesis_pca_tsne.png")
'''
# FIGURE 1: Domain separation (Original vs DCT)
# NO class coloring
'''
plt.figure(figsize=(8,6))

plt.scatter(
    Z[domain==0, 0], Z[domain==0, 1],
    color="magenta", alpha=0.5, label="Original"
)

plt.scatter(
    Z[domain==1, 0], Z[domain==1, 1],
    color="gold", alpha=0.5, label="DCT"
)

plt.legend()
plt.title("t-SNE: Feature Distribution (Original vs DCT)")
plt.tight_layout()
plt.savefig("tsne_domain.png", dpi=300)
plt.close()
'''
# FIGURE 2: Class separation (Real vs Fake)
'''
plt.figure(figsize=(8,6))

plt.scatter(
    Z[y==0, 0], Z[y==0, 1],
    color="green", alpha=0.5, label="Real"
)

plt.scatter(
    Z[y==1, 0], Z[y==1, 1],
    color="red", alpha=0.5, label="Fake"
)

plt.legend()
plt.title("t-SNE: Real vs Fake Separation (Combined)")
plt.tight_layout()
plt.savefig("tsne_class.png", dpi=300)
plt.close()
'''
# FIGURE 3 (BEST, recommended): Class separation within each model
# Original model only

plt.figure(figsize=(8,6))

idx = domain == 0

plt.scatter(
    Z[idx & (y == 0), 0], Z[idx & (y == 0), 1],
    color="green", alpha=0.5, label="Real"
)

plt.scatter(
    Z[idx & (y == 1), 0], Z[idx & (y == 1), 1],
    color="red", alpha=0.5, label="Fake"
)

plt.legend()
plt.title("Original Model: Real vs Fake")
plt.savefig("tsne_orig_class.png", dpi=300)
plt.close()


# DCT model only
'''
plt.figure(figsize=(8,6))
idx = domain == 1

plt.scatter(Z[idx & (y==0), 0], Z[idx & (y==0), 1],
            color="green", alpha=0.5, label="Real")
plt.scatter(Z[idx & (y==1), 0], Z[idx & (y==1), 1],
            color="red", alpha=0.5, label="Fake")

plt.legend()
plt.title("DCT Model 4th order: Real vs Fake")
plt.savefig("tsne_dct_class.png", dpi=300)
plt.close()
'''

# ==========================================================
# FINAL PLOT: Domain (shape) + Class (color) together
# ==========================================================
'''
plt.figure(figsize=(8,6))

# ---- Original domain (square) ----
plt.scatter(
    Z[(domain == 0) & (y == 0), 0],
    Z[(domain == 0) & (y == 0), 1],
    marker='s', color='green', alpha=0.6,
    label='Original - Real'
)

plt.scatter(
    Z[(domain == 0) & (y == 1), 0],
    Z[(domain == 0) & (y == 1), 1],
    marker='s', color='red', alpha=0.6,
    label='Original - Fake'
)

# ---- DCT domain (triangle) ----
plt.scatter(
    Z[(domain == 1) & (y == 0), 0],
    Z[(domain == 1) & (y == 0), 1],
    marker='^', color='green', alpha=0.6,
    label='DCT - Real'
)

plt.scatter(
    Z[(domain == 1) & (y == 1), 0],
    Z[(domain == 1) & (y == 1), 1],
    marker='^', color='red', alpha=0.6,
    label='DCT - Fake'
)

plt.legend()
plt.title("t-SNE: Domain (Shape) + Class (Color)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.savefig("tsne_domain_class_combined.png", dpi=300)
plt.close()

print("Saved plot: tsne_domain_class_combined.png")
'''
