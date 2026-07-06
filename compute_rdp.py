import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

ensemble = np.load("ensemble_1000.npy")
std_map = np.std(ensemble, axis=0)[:, 120:]

# T = gap / 2 = 0.10 / 2 = 0.05
GAP = 0.10
THRESHOLD = GAP / 2  # = 0.05

std_masked = np.where(std_map <= THRESHOLD, std_map, np.nan)

# RDP: for each row find the first NaN column (sentinel ensures fully-valid rows
# resolve to n_cols). Take the max across all rows = furthest distance.
nan_mask = np.isnan(std_masked)
padded = np.hstack([nan_mask, np.ones((nan_mask.shape[0], 1), dtype=bool)])
first_nan = np.argmax(padded, axis=1)
rdp = int(np.max(first_nan))
print(f"RDP = {rdp} (T={THRESHOLD}, gap={GAP})")

plt.figure(figsize=(10, 5))
plt.imshow(std_masked, aspect="auto", cmap="viridis", vmin=0, vmax=std_map.max())
plt.colorbar(label="Standard deviation")
plt.axvline(rdp, color="black", linestyle="--", linewidth=1.5, label=f"RDP = {rdp}")
plt.title(f"Per-pixel std  (T = gap/2 = {THRESHOLD})  —  RDP = {rdp}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rdp.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUTDIR, "rdp.pdf"), bbox_inches="tight")
print("Saved rdp.png and rdp.pdf to figures/")
plt.show()