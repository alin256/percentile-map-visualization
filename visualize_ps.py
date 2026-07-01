import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors

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

over = percentiles > 95
under = percentiles < 5

plt.figure("Over", figsize=(10, 6))
# plt.imshow(ensemble[0])
plt.imshow(over, aspect='auto')

plt.figure("Under", figsize=(10, 6))
plt.imshow(under, aspect='auto')

plt.figure("Percentiles", figsize=(10, 6))
plt.imshow(percentiles, aspect='auto', 
           cmap="tab20b", 
           vmin=0,
           vmax=100
           )
plt.colorbar()


sorted_percentiles = np.sort(percentiles.ravel())
n = len(sorted_percentiles)
theoretical = np.linspace(0, 100, n)

COLORMAP = "cividis"

plt.figure("Sorted Percentiles", figsize=(10, 6))

n_cols = percentiles.shape[1]

highlighted = list(range(6)) + [10] + list(range(20, n_cols, 20))

n_colors = len(highlighted)

colors = matplotlib.colormaps[COLORMAP](np.linspace(0, 1, n_cols))


plt.figure("Sorted Percentiles", figsize=(10, 6))

for col_idx_hl in highlighted:
    col_idx = highlighted[col_idx_hl]
    col_sorted = np.sort(percentiles[:, col_idx])
    x = np.linspace(0, 1, len(col_sorted))
    if col_idx in highlighted:
        plt.plot(x, col_sorted, 
                 color=colors[col_idx_hl], 
                 #  alpha=0.9, 
                 linewidth=0.5, label=f"Column {col_idx}")
    else:
        pass
        # plt.plot(x, col_sorted, color=colors[col_idx], alpha=0.2, linewidth=0.5)

n_all = len(sorted_percentiles)
plt.plot(np.linspace(0, 1, n_all), sorted_percentiles, color="black", linewidth=1.5, label="All pixels")
plt.plot([0, 1], [0, 100], label="Theoretical (uniform)", linestyle="--", color="red", linewidth=1.5)
plt.xlabel("Rank (normalized)")
plt.ylabel("Percentile")
plt.title("Sorted percentiles — one curve per column")
plt.legend(fontsize=7, ncol=2)
plt.grid(True)

sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=matplotlib.colors.Normalize(vmin=0, vmax=n_cols - 1))
plt.colorbar(sm, ax=plt.gca(), label="Column index (left → right)")

plt.show()