"""ML pipeline for productivity classification."""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    return pd.read_csv(file_path)

def preprocess_features(df, target_col):
    # Ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset!")

    X = df.drop(columns=[target_col], errors="ignore")  # safe drop
    y = df[target_col]

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Encode target if categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train Productivity Classifier")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="productivity_class", help="Target column")
    parser.add_argument("--save_model", type=str, default="productivity_classifier.pkl", help="Output model file")
    args = parser.parse_args()

    # Load data
    df = load_data(args.data)

    # Preprocess features
    X, y = preprocess_features(df, args.target)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save model AND feature names
    model_info = {"model": clf, "feature_names": X.columns.tolist()}
    joblib.dump(model_info, args.save_model)
    print(f"✅ Model saved to {args.save_model}")

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

