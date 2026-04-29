import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load dataset Titanic
import seaborn as sns
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Handle missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Encode kategori
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Split
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Evaluasi
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    }

# Single Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
result_dt = evaluate_model(dt, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
result_rf = evaluate_model(rf, "Random Forest")

# Gradien Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
result_gb = evaluate_model(gb, "Gradient Boosting")

# Stacking
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=50))
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

result_stack = evaluate_model(stack, "Stacking")

# Menampilkan Hasil
results = pd.DataFrame([result_dt, result_rf, result_gb, result_stack])
print(results)