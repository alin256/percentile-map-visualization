import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

std_map = np.load(os.path.join(OUTDIR, "std_map.npy"))

GAP = 0.10
N = 2
threshold = GAP / N  # T = 0.05

std_masked = np.where(std_map <= threshold, std_map, np.nan)

nan_mask = np.isnan(std_masked)
padded = np.hstack([nan_mask, np.ones((nan_mask.shape[0], 1), dtype=bool)])
first_nan = np.argmax(padded, axis=1)
rdp = int(np.max(first_nan))
pct_kept = 100 * np.sum(~nan_mask) / std_map.size
print(f"RDP = {rdp}  T={threshold}  ({pct_kept:.1f}% kept)")

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(std_masked, aspect="auto", cmap="viridis", vmin=0, vmax=std_map.max())
ax.axvline(rdp, color="black", linestyle="--", linewidth=1., label=f"Reliable Distance of Prediction = {rdp} columns")
ax.set_title(f"Ensemble std — low-uncertainty region (std $\\leq$ {threshold})  |  RDP = {rdp} columns")
ax.legend()
fig.colorbar(im, ax=ax, label="Standard deviation")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "std_map_reference.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTDIR, "std_map_reference.pdf"), bbox_inches="tight")
print("Saved std_map.png and std_map.pdf to figures/")
plt.show()

