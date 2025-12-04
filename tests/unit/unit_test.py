import pytest
from model_pipeline import clean_data


def test_clean_data_returns_dataframe():
    import pandas as pd

    df = pd.DataFrame(
        {
            "department": ["A", "B"],
            "quarter": ["Q1", "Q2"],
            "day": ["Monday", "Tuesday"],
            "wip": [10, 20],
            "over_time": [1, 2],
            "idle_time": [0, 1],
            "no_of_workers": [2, 4],
            "idle_men": [0, 1],
            "actual_productivity": [50, 60],
        }
    )
    cleaned_df = clean_data(df)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert not cleaned_df.empty
