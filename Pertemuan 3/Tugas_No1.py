import numpy as np
from scipy.spatial.distance import euclidean, cityblock

# Data contoh
P = np.array([1, 2, 3])
Q = np.array([4, 5, 6])

# Perhitungan berbagai jarak
euc_dist = euclidean(P, Q)
man_dist = cityblock(P, Q)
print(f"Euclidean Distance: {euc_dist:.2f}")
print(f"Manhattan Distance: {man_dist:.2f}")