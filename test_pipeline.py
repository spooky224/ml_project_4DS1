# test_pipeline.py

from model_pipeline import (
    load_data,
    clean_data,
    feature_engineering,
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    load_and_evaluate_model,
)


def main():
    """Full pipeline: load, clean, feature-engineer, train, evaluate, save."""
    # Step 1: Load data
    df = load_data("productivityPrediction.csv")

    # Step 2: Clean data
    df_clean = clean_data(df)

    # Step 3: Feature engineering
    df_feat, X_cols, target = feature_engineering(df_clean)

    # Step 4: Preprocess + split
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df_feat, X_cols, target
    )

    # Step 5: Train model
    model = train_model(X_train, y_train, preprocessor)

    # Step 6: Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Step 7: Save model
    save_model(model, "lr_productivity_model.pkl")

    # Step 8: Load model and evaluate again
    loaded_model, metrics_loaded, comparison_df = load_and_evaluate_model(
        "lr_productivity_model.pkl", X_test, y_test
    )

    print("First 10 predictions vs actual:")
    print(comparison_df.head(10))
    print("Loaded model metrics:", metrics_loaded)


if __name__ == "__main__":
    main()
