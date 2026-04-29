import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Buat dan latih model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Visualisasi pohon keputusan
plt.figure(figsize=(15, 10))
plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree - Iris Dataset")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Interpretasi: node menampilkan
# - samples: jumlah data di node
# - value: [jumlah kelas 0, kelas 1, kelas 2]
# - class: kelas mayoritas
# - gini/entropy: impurity

# Feature importance
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance)