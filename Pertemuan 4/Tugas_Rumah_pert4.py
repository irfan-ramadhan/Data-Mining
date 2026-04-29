import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Decision Tree dari Scratch
class DecisionTreeScratch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def information_gain(self, X_column, y, threshold):
        parent_entropy = self.entropy(y)

        left_idx = X_column <= threshold
        right_idx = X_column > threshold

        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])

        e_left = self.entropy(y[left_idx])
        e_right = self.entropy(y[right_idx])

        child_entropy = (n_left/n)*e_left + (n_right/n)*e_right

        return parent_entropy - child_entropy

    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1:
            return Counter(y).most_common(1)[0][0]

        feature_idx, threshold = self.best_split(X, y)

        if feature_idx is None:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return {"feature": feature_idx, "threshold": threshold, "left": left, "right": right}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree["feature"]] <= tree["threshold"]:
            return self.predict_sample(x, tree["left"])
        else:
            return self.predict_sample(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
# Training dan Evaluasi Scratch
tree_scratch = DecisionTreeScratch(max_depth=3)
tree_scratch.fit(X_train, y_train)

y_pred_scratch = tree_scratch.predict(X_test)
accuracy_scratch = np.mean(y_pred_scratch == y_test)

print("Akurasi Decision Tree (Scratch):", accuracy_scratch)

# Menggunakan Scikit-learn
tree_sklearn = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_sklearn.fit(X_train, y_train)

y_pred_sklearn = tree_sklearn.predict(X_test)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test)

print("Akurasi Decision Tree (Scikit-learn):", accuracy_sklearn)

# Perbandingan Hasil
print("\nPerbandingan:")
print(f"Scratch     : {accuracy_scratch:.4f}")
print(f"Scikit-learn: {accuracy_sklearn:.4f}")