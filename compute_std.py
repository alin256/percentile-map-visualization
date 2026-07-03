import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

ensemble = np.load("ensemble_1000.npy")
print(f"ensemble shape: {ensemble.shape}, dtype: {ensemble.dtype}")

# Compute per-pixel standard deviation across ensemble members (axis 0)
std_map = np.std(ensemble, axis=0)

std_map = std_map[:,120:]

mean_std = std_map.mean()
print(f"std_map shape: {std_map.shape}")
print(f"std min: {std_map.min():.6f}, max: {std_map.max():.6f}, mean: {mean_std:.6f}")

# Three object classes with mean property values: 0.02, 0.12, 0.20
# Pairwise gaps:
#   smallest adjacent  = 0.08  (0.12 vs 0.20)
#   largest adjacent   = 0.10  (0.02 vs 0.12)
#   full range         = 0.18  (0.02 vs 0.20)
#
# Confidence levels (z-score factor k, threshold = gap / (2*k)):
#   99.7%  →  k = 3.000
#   95%    →  k = 2.000
#   Q1-Q3  →  k = 0.6745
#
# All 9 combinations (pixels with std > threshold are NaN'd out):
gaps = {
    "gap=0.08 (close pair)":  0.08,
    "gap=0.10 (far pair)":    0.10,
    "gap=0.18 (full range)":  0.18,
}
confidence = {
    "99.7% (±3σ)":   3.000,
    "95%   (±2σ)":   2.000,
    "Q1–Q3 (±0.67σ)": 0.6745,
}

np.save(os.path.join(OUTDIR, "std_map.npy"), std_map)

fig, axes = plt.subplots(
    len(confidence), len(gaps),
    figsize=(5 * len(gaps), 4 * len(confidence)),
    squeeze=False,
)

vmax = std_map[~np.isnan(std_map)].max()

for row, (conf_label, k) in enumerate(confidence.items()):
    for col, (gap_label, gap) in enumerate(gaps.items()):
        threshold = gap / (2 * k)
        std_masked = np.where(std_map <= threshold, std_map, np.nan)
        ax = axes[row][col]
        im = ax.imshow(std_masked, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        pct_kept = 100 * np.sum(~np.isnan(std_masked)) / std_map.size
        ax.set_title(
            f"{conf_label}\n{gap_label}\nT={threshold:.4f}  ({pct_kept:.1f}% kept)",
            fontsize=9,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Per-pixel std masked by threshold — all confidence × gap combinations", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "std_map.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTDIR, "std_map.pdf"), bbox_inches="tight")
print("Saved std_map.png and std_map.pdf to figures/")
