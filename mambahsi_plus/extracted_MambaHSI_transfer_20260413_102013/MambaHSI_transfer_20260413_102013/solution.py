import numpy as np
import os

def print_info(name, data):
    unique, counts = np.unique(data, return_counts=True)
    print(f"--- {name} ---")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Unique values: {unique}")
    print(f"Counts: {dict(zip(unique, counts))}")
    print(f"Total elements: {data.size}")

# 1) data/label/label1_gt.npy
label1_gt = np.load('data/label/label1_gt.npy')
print_info("label1_gt.npy", label1_gt)

# 2) data/label2/label2.enp
# The expected size was 1024 * 5029 = 5149696, but size is 1715923.
# 1715923 / 5029 is approx 341.1...
# Let's read it as is and check stats without reshape if reshape fails.
try:
    data = np.fromfile('data/label2/label2.enp', dtype=np.uint8)
    print("--- label2.enp ---")
    print(f"Actual data size: {data.size}")
    print(f"Dtype: {data.dtype}")
    unique, counts = np.unique(data, return_counts=True)
    print(f"Unique values: {unique}")
    print(f"Counts: {dict(zip(unique, counts))}")
    # Still print shape if possible or reason why not
    print(f"Shape: {data.shape} (Could not reshape to 5029x1024)")
except Exception as e:
    print(f"Error reading label2.enp: {e}")

# 3) data/label/label1_cube.npy shape
label1_cube = np.load('data/label/label1_cube.npy', mmap_mode='r')
print(f"--- label1_cube.npy ---")
print(f"Shape: {label1_cube.shape}")
