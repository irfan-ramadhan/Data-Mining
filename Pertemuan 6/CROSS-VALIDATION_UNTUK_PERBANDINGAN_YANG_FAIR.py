from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import time
import pandas as pd

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Definisi Stacking
estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# List model yang akan dibandingkan
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Stacking': stacking
}

# Cross-validation dengan 5-fold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    elapsed_time = time.time() - start_time

    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std': scores.std(),
        'Training Time (s)': elapsed_time
    })

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))

# Kesimpulan
best_model = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'Model']
print(f"\nModel terbaik berdasarkan CV: {best_model}")