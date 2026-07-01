import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"{name}.pdf"), bbox_inches="tight")

ensemble = np.load("ensemble_1000.npy")
print(type(ensemble))
print(ensemble.shape)
print(ensemble.dtype)
print(f"ensemble min: {ensemble.min()}, max: {ensemble.max()}")


# img = np.array(Image.open("patch_1088.png"))/256.
img = np.load("patch_1088.npy")
print(f"img min: {img.min()}, max: {img.max()}")


print(img.shape)
print(img.dtype)

a2 = img
a3 = ensemble

# the <= outputs 0 or 1 thus mean gives percentage <=
percentiles = 100. * np.mean(a3 <= a2[None, :, :], axis=0)

print(percentiles.shape)

percentiles = percentiles[:,120:]
percentiles_crop = percentiles[:,0:15]

over = percentiles > 95
under = percentiles < 5

plt.figure("Over", figsize=(10, 6))
# plt.imshow(ensemble[0])
plt.imshow(over, aspect='auto')
plt.title("Pixels above 95th percentile")
save_fig("over")

plt.figure("Under", figsize=(10, 6))
plt.imshow(under, aspect='auto')
plt.title("Pixels below 5th percentile")
save_fig("under")

plt.figure("Percentiles", figsize=(10, 6))
plt.imshow(percentiles_crop, aspect='auto', 
        #    cmap="tab20b", 
           vmin=0,
           vmax=100
           )
plt.colorbar()
plt.title("Percentile map")
save_fig("percentiles")


sorted_percentiles = np.sort(percentiles.ravel())
n = len(sorted_percentiles)
theoretical = np.linspace(0, 100, n)

COLORMAP = "coolwarm"

plt.figure("Sorted Percentiles", figsize=(10, 6))

n_cols = percentiles.shape[1]

highlighted = list(range(6)) + [10] + list(range(25, n_cols, 25))

n_colors = len(highlighted)

colors = matplotlib.colormaps[COLORMAP](np.linspace(0, 1, n_colors))


plt.figure("Sorted Percentiles", figsize=(10, 6))

small_line_width = .8

for i, col_idx in enumerate(highlighted):
    col_sorted = np.sort(percentiles[:, col_idx])
    x = np.linspace(0, 1, len(col_sorted))
    if i == 0 or i == 5:
        plt.plot(x, col_sorted, 
            color=colors[i], 
            linewidth=small_line_width, 
            label=f"Column {col_idx}")
    else:
        plt.plot(x, col_sorted, 
            color=colors[i], 
            linewidth=small_line_width)


n_all = len(sorted_percentiles)
plt.plot(np.linspace(0, 1, n_all), sorted_percentiles, color="black", linewidth=1.5, label="All pixels")
plt.plot([0, 1], [0, 100], label="Theoretical (uniform)", linestyle="--", color="red", linewidth=1.5)
plt.xlabel("Rank (normalized)")
plt.ylabel("Percentile")
plt.title("Sorted percentiles — one curve per column")
plt.legend(ncol=2)
plt.grid(True)

norm = matplotlib.colors.Normalize(vmin=0, vmax=n_colors - 1)
sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
cbar = plt.colorbar(sm, ax=plt.gca(), label="Column index")
cbar.set_ticks(np.linspace(0, n_colors - 1, n_colors))
cbar.set_ticklabels([str(c) for c in highlighted])
save_fig("sorted_percentiles")

plt.show()