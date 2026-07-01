import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

plt.figure("Sorted Percentiles", figsize=(10, 6))
plt.plot(sorted_percentiles, label="Observed")
plt.plot(theoretical, label="Theoretical (uniform)", linestyle="--", color="red")
plt.xlabel("Pixel rank")
plt.ylabel("Percentile")
plt.title("Sorted percentiles across all pixels")
plt.legend()
plt.grid(True)

plt.show()