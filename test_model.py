# test_model.py

import argparse
from model_pipeline import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_data,
    evaluate_model,
    load_model,
)


def main():
    """Load a saved model and evaluate it on a dataset."""
    parser = argparse.ArgumentParser(description="Load a saved model and evaluate it")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--model_file", type=str, required=True, help="Path to saved .pkl model"
    )
    parser.add_argument(
        "--target", type=str, default="actual_productivity", help="Target column"
    )
    args = parser.parse_args()

    # Load dataset
    df = load_data(args.data)
    df = clean_data(df, target=args.target)
    df, X_cols, target = feature_engineering(df, target=args.target)

    # Preprocess + split (only to get X_test, y_test)
    _, X_test, _, y_test, _ = preprocess_data(df, X_cols, target)

    # Load saved model
    model = load_model(args.model_file)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    print("\n\u2705 Loaded Model Evaluation Complete!")
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()
