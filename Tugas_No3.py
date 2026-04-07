import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# IMPLEMENTASI JARAK NUMERIK PADA DATASET IRIS
iris = load_iris()
iris_data = iris.data
feature_names = iris.feature_names
df = pd.DataFrame(iris_data, columns=feature_names)
print("==DATASET IRIS==")
print(df.head())

# NORMALISASI DATA
scaler = MinMaxScaler()
iris_normalized = scaler.fit_transform(iris_data)
data = iris_normalized
print("\nDATA SETELAH NORMALISASI")
print(pd.DataFrame(data, columns=feature_names).head())

# CONTOH PERHITUNGAN JARAK
A = data[0]
B = data[1]
euc_dist = euclidean(A, B)
man_dist = cityblock(A, B)
min_dist = minkowski(A, B, p=3)
print("\nCONTOH PERHITUNGAN DATA KE-0 DAN DATA KE-1")
print(f"Euclidean Distance : {euc_dist:.4f}")
print(f"Manhattan Distance : {man_dist:.4f}")
print(f"Minkowski Distance : {min_dist:.4f}")

# LOOPING JARAK 5 DATA PERTAMA
print("\n==PERHITUNGAN JARAK 5 DATA PERTAMA==")
for i in range(5):
    for j in range(i + 1, 5):
        euc_dist = euclidean(data[i], data[j])
        man_dist = cityblock(data[i], data[j])
        min_dist = minkowski(data[i], data[j], p=3)
        print(f"\nData ke-{i} dengan Data ke-{j}")
        print(f"Euclidean Distance : {euc_dist:.4f}")
        print(f"Manhattan Distance : {man_dist:.4f}")
        print(f"Minkowski Distance : {min_dist:.4f}")

# MATRIKS JARAK
n = len(data)
euclidean_matrix = np.zeros((n, n))
manhattan_matrix = np.zeros((n, n))
minkowski_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        euclidean_matrix[i][j] = euclidean(data[i], data[j])
        manhattan_matrix[i][j] = cityblock(data[i], data[j])
        minkowski_matrix[i][j] = minkowski(data[i], data[j], p=3)
print("\n==MATRIKS JARAK EUCLIDEAN (5x5)==")
print(pd.DataFrame(euclidean_matrix[:5, :5]).round(4))
print("\n==MATRIKS JARAK MANHATTAN (5x5)==")
print(pd.DataFrame(manhattan_matrix[:5, :5]).round(4))
print("\n==MATRIKS JARAK MINKOWSKI (5x5)==")
print(pd.DataFrame(minkowski_matrix[:5, :5]).round(4))

# COSINE SIMILARITY DENGAN TF-IDF
documents = [
    "data science is fun",
    "data mining is fun",
    "machine learning is cool"
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("\nVOCABULARY")
print(vectorizer.get_feature_names_out())
print("\nTF-IDF MATRIX")
print(tfidf_matrix.toarray().round(4))

cos_sim = cosine_similarity(tfidf_matrix)
print("\nCOSINE SIMILARITY MATRIX")
print(cos_sim.round(4))
print("\nCosine Similarity D1-D2 :", round(cos_sim[0][1], 4))
print("Cosine Similarity D1-D3 :", round(cos_sim[0][2], 4))