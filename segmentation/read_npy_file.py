import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG: Set your path to a .npy file ===
file_path = "../datasets/kitti/sequences_segmentation/00/image_2/segmentclasses/000123.npy"

# === 1. Load the .npy segmentation map ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

seg = np.load(file_path)

# === 2. Print useful debug info ===
print(f"Shape: {seg.shape}")
print(f"Dtype: {seg.dtype}")
print(f"Unique class IDs: {np.unique(seg)}")

# === 3. Visualize it with matplotlib ===
plt.figure(figsize=(8, 6))
plt.imshow(seg, cmap="tab20")
plt.colorbar()
plt.title("Segment Class Map")
plt.show()