"""Main module to run the ML pipeline for productivity prediction."""

import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from model_pipeline import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
)


def main():
    """Main function to execute the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Productivity Prediction"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--target", type=str, default="actual_productivity", help="Target column"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "rf"],
        help="Model type",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="productivity_model.pkl",
        help="Filename for saved model",
    )
    args = parser.parse_args()

    # Load and clean data
    data_frame = load_data(args.data)
    data_frame = clean_data(data_frame, target=args.target)

    # Feature engineering
    data_frame, feature_columns, target_column = feature_engineering(
        data_frame, target=args.target
    )

    # Choose model
    if args.model == "linear":
        model_instance = LinearRegression()
    else:
        model_instance = RandomForestRegressor(n_estimators=100, random_state=42)

    # Preprocess and split
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data(
        data_frame, feature_columns, target_column
    )

    # Train
    pipeline_model = train_model(x_train, y_train, preprocessor, model_instance)

    # Evaluate
    metrics = evaluate_model(pipeline_model, x_test, y_test)

    # Save
    save_model(pipeline_model, args.save_model)

    print("\u2705 Pipeline completed successfully!")
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()
