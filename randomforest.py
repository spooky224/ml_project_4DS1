import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# ---------------- Load dataset ----------------
df = pd.read_csv("productivityPrediction.csv")

# ---------------- Clean dataset ----------------
df = df.dropna(subset=["actual_productivity"])  # drop rows without productivity

# Clean categorical columns
for col in ["quarter", "department", "day"]:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace(['nan', 'none', '', '0'], 'unknown')

# ---------------- Bin continuous productivity into 3 equal-interval classes ----------------
num_bins = 3
df["productivity_class"] = pd.cut(
    df["actual_productivity"], bins=num_bins, labels=False, include_lowest=True
)

# ---------------- Define features and target ----------------
X = df.drop(columns=["productivity_class", "actual_productivity", "date"])
y = df["productivity_class"].astype(int)

# ---------------- Fill numeric missing values ----------------
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(0)

# ---------------- Feature Engineering ----------------
X["efficiency"] = X["targeted_productivity"] / X["smv"].replace(0, 1)
X["idle_per_worker"] = X["idle_time"] / X["idle_men"].replace(0, 1)
X["wip_per_worker"] = X["wip"] / X["no_of_workers"].replace(0, 1)
X["overtime_ratio"] = X["over_time"] / X["team"].replace(0, 1)

# ---------------- Encode categorical features ----------------
label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"{col} allowed values: {list(le.classes_)}")

# ---------------- Stratified Train/Test Split ----------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---------------- Train RandomForest classifier ----------------
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train, y_train)

print("\nRandomForest Classifier trained successfully!")
print("Test Accuracy:", clf.score(X_test, y_test))

# ---------------- Save classifier for GUI ----------------
joblib.dump({
    "model": clf,
    "feature_names": X.columns.tolist(),
    "label_encoders": label_encoders
}, "productivity_classifier.pkl")

# ---------------- Mapping numeric classes to human-readable ranges ----------------
class_mapping = {}
intervals = pd.cut(df["actual_productivity"], bins=num_bins, retbins=True)[1]
for i in range(num_bins):
    class_mapping[i] = f"{intervals[i]:.3f} - {intervals[i+1]:.3f}"

print("\nClass mapping:")
for k, v in class_mapping.items():
    print(f"Class {k}: Productivity range {v}")

# ---------------- Sample predictions ----------------
print("\n--- Sample Predictions with Actual Classes ---")

# Random 10-sample selection each run
df_samples = df.sample(10).copy()
df_samples_input = df_samples.drop(columns=["productivity_class", "actual_productivity", "date"])

correct_count = 0
wrong_count = 0

for i, row in df_samples_input.iterrows():
    df_sample = pd.DataFrame([row])

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df_sample.columns:
            val = str(df_sample.iloc[0][col]).strip().lower()
            if val not in le.classes_:
                val = 'unknown'
            df_sample[col] = le.transform([val])

    # Ensure all classifier features exist
    for col in X.columns:
        if col not in df_sample.columns:
            df_sample[col] = 0

    df_sample = df_sample[X.columns]

    # Predict
    pred_class = clf.predict(df_sample)[0]
    pred_proba = clf.predict_proba(df_sample)[0]
    actual_class = df_samples.loc[i, "productivity_class"]  # fixed index issue
    pred_range = class_mapping[pred_class]
    actual_range = class_mapping[int(actual_class)]
    correct = "\u2705" if pred_class == actual_class else "\u274c"

    print(f"Sample {i}: Predicted Class {pred_class} ({pred_range}) | "
          f"Actual Class {int(actual_class)} ({actual_range}) {correct}")
    print("Class probabilities:", {cls: f"{p:.2f}" for cls, p in enumerate(pred_proba)}, "\n")

    # Quick report counting
    if pred_class == actual_class:
        correct_count += 1
    else:
        wrong_count += 1

print("\n--- Prediction Report ---")
print(f"Total samples evaluated: {len(df_samples_input)}")
print(f"Correct predictions: {correct_count} \u2705")
print(f"Wrong predictions: {wrong_count} \u274c")
print(f"Accuracy on sample: {correct_count / len(df_samples_input) * 100:.2f}%")

# ---------------- Feature Importance ----------------
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

