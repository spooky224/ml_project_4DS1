"""Pipeline module for data preprocessing, feature engineering, and ML model training."""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(filename: str, sep: str = ",") -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    try:
        data_frame = pd.read_csv(filename, sep=sep)
        logging.info("Successfully loaded %s", filename)
        return data_frame
    except Exception as ex:
        logging.error("Error loading file: %s", ex)
        return None


def explore_data(data_frame: pd.DataFrame, num_rows: int = 5):
    """Display basic info, head, missing values, and unique categorical values."""
    if data_frame is None:
        logging.warning("No DataFrame provided")
        return

    logging.info("Dataset shape: %s", data_frame.shape)
    logging.info("Dataset head:\n%s", data_frame.head(num_rows))
    data_frame.info()
    logging.info("Missing values per column:\n%s", data_frame.isnull().sum())
    logging.info("Summary statistics:\n%s", data_frame.describe())
    for col in ["day", "department", "quarter"]:
        if col in data_frame.columns:
            logging.info("Unique values in '%s': %s", col, data_frame[col].unique())


def clean_data(
    data_frame: pd.DataFrame, target: str = "actual_productivity"
) -> pd.DataFrame:
    """Clean dataset by handling NaNs, fixing unexpected values, and dropping missing target rows."""
    for col in ["department", "quarter", "day"]:
        if col in data_frame.columns:
            data_frame[col] = data_frame[col].astype(str).str.strip()

    if "quarter" in data_frame.columns:
        data_frame["quarter"] = data_frame["quarter"].replace({"Quarter5": "Quarter4"})
    if "date" in data_frame.columns:
        data_frame["date"] = pd.to_datetime(data_frame["date"], errors="coerce")

    numeric_cols = data_frame.select_dtypes(include=np.number).columns.tolist()
    if "wip" in numeric_cols:
        numeric_cols.remove("wip")
    for col in numeric_cols:
        data_frame[col].fillna(data_frame[col].mean(), inplace=True)

    for col in ["day", "department", "quarter"]:
        if col in data_frame.columns:
            data_frame[col] = data_frame[col].replace("nan", np.nan)
            if not data_frame[col].isnull().all():
                data_frame[col].fillna(data_frame[col].mode()[0], inplace=True)

    if "wip" in data_frame.columns:
        data_frame["wip"].fillna(data_frame["wip"].median(), inplace=True)

    if target in data_frame.columns:
        data_frame = data_frame.dropna(subset=[target])

    logging.info("Data cleaned. New shape: %s", data_frame.shape)
    return data_frame


def feature_engineering(data_frame: pd.DataFrame, target: str = "actual_productivity"):
    """
    Add per-worker features and define feature columns.
    Returns (data_frame, feature_columns, target_column)
    """
    required_cols = ["wip", "over_time", "idle_time", "no_of_workers", "idle_men"]
    for col in required_cols:
        if col not in data_frame.columns:
            raise ValueError("Required column '%s' not found", col)

    data_frame["wip_per_worker"] = data_frame["wip"] / data_frame[
        "no_of_workers"
    ].replace(0, np.nan)
    data_frame["over_time_per_worker"] = data_frame["over_time"] / data_frame[
        "no_of_workers"
    ].replace(0, np.nan)
    data_frame["idle_time_per_worker"] = data_frame["idle_time"] / data_frame[
        "idle_men"
    ].replace(0, np.nan)

    data_frame[["wip_per_worker", "over_time_per_worker", "idle_time_per_worker"]] = (
        data_frame[
            ["wip_per_worker", "over_time_per_worker", "idle_time_per_worker"]
        ].fillna(0)
    )

    feature_columns = [col for col in data_frame.columns if col not in [target, "date"]]

    logging.info(
        "Feature engineering done. Example features:\n%s",
        data_frame[
            ["wip_per_worker", "over_time_per_worker", "idle_time_per_worker"]
        ].head(),
    )
    return data_frame, feature_columns, target


def preprocess_data(
    data_frame: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    categorical_cols: list = None,
):
    """Prepare preprocessing pipeline and train/test split."""
    if categorical_cols is None:
        categorical_cols = [
            col for col in ["day", "department", "quarter"] if col in data_frame.columns
        ]

    numeric_cols = [col for col in feature_columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )

    x_data = data_frame[feature_columns]
    y_data = data_frame[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    logging.info(
        "Preprocessing + train/test split done. Train shape: %s, Test shape: %s",
        x_train.shape,
        x_test.shape,
    )
    return x_train, x_test, y_train, y_test, preprocessor


def train_model(x_train, y_train, preprocessor, model=None):
    """Train a model using a preprocessing pipeline. Default model: LinearRegression."""
    if model is None:
        model = LinearRegression()

    pipeline_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline_model.fit(x_train, y_train)
    logging.info("Model training completed.")
    return pipeline_model


def evaluate_model(model, x_test, y_test) -> dict:
    """Evaluate model performance and return RMSE, MAE, and R2 metrics."""
    y_pred = model.predict(x_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    logging.info(
        "Model evaluation -> RMSE: %.4f, MAE: %.4f, R2: %.4f",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
    )
    return metrics


def save_model(model, filename: str, version: bool = True):
    """Save model to disk with optional timestamp versioning."""
    if version:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{timestamp}{ext}"
    joblib.dump(model, filename)
    logging.info("Model saved as '%s'", filename)


def load_model(filename: str):
    """Load a saved model from disk."""
    model = joblib.load(filename)
    logging.info("Model loaded from '%s'", filename)
    return model


def load_and_evaluate_model(filename: str, x_test, y_test):
    """Load a model from disk and evaluate it on a test set."""
    model = load_model(filename)
    y_pred = model.predict(x_test)
    comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    logging.info("First 10 predictions:\n%s", comparison_df.head(10))
    metrics = evaluate_model(model, x_test, y_test)
    return model, metrics, comparison_df
