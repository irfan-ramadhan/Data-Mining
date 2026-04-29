import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset Iris
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Implementasi k-NN dari Scratch
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []

    for test_point in X_test:
        distances = []

        # Hitung jarak ke semua data training
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))

        # Urutkan berdasarkan jarak
        distances.sort(key=lambda x: x[0])

        # Ambil k tetangga terdekat
        k_neighbors = distances[:k]
        k_labels = [label for _, label in k_neighbors]

        # Voting mayoritas
        pred = max(set(k_labels), key=k_labels.count)
        predictions.append(pred)

    return np.array(predictions)

# Uji berbagai nilai k
k_values = [1, 3, 5, 7, 9, 11]

print("=== HASIL PERBANDINGAN ===\n")

for k in k_values:
    # Scratch
    y_pred_scratch = knn_predict(X_train, y_train, X_test, k)
    acc_scratch = np.mean(y_pred_scratch == y_test)

    # Scikit-learn
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_sklearn = model.predict(X_test)
    acc_sklearn = np.mean(y_pred_sklearn == y_test)

    print(f"k = {k}")
    print(f"  Scratch     : {acc_scratch:.4f}")
    print(f"  Scikit-learn: {acc_sklearn:.4f}")
    print()