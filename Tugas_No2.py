import numpy as np
from sklearn.metrics import jaccard_score

def simple_matching_coefficient(x, y):
    """Menghitung SMC untuk data vektor biner"""
    x = np.array(x)
    y = np.array(y)

    m11 = np.sum((x == 1) & (y == 1))
    m00 = np.sum((x == 0) & (y == 0))
    total = len(x)

    return (m11 + m00) / total

# Data contoh
x = [1, 0, 1, 0, 1, 1]
y = [1, 1, 0, 0, 1, 0]
# Hitung SMC
smc = simple_matching_coefficient(x, y)
print(f"SMC: {smc:.2f}") 
# Hitung Jaccard (hanya kehadiran)
# M11 = 2, M10 = 2, M01 = 1
m11 = np.sum((np.array(x) == 1) & (np.array(y) == 1))
m10 = np.sum((np.array(x) == 1) & (np.array(y) == 0))
m01 = np.sum((np.array(x) == 0) & (np.array(y) == 1))

jaccard = m11 / (m11 + m10 + m01)
print(f"Jaccard (manual): {jaccard:.2f}")

# Alternatif dengan scikit-learn (average='binary')
jaccard_sklearn = jaccard_score(x, y, average ='binary')
print(f"Jaccard (sklearn): {jaccard_sklearn:2f}")