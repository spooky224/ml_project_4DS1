# validate_data.py
import pandas as pd
import sys

DATA_FILE = "productivityPrediction.csv"

def validate_data(df):
    errors = []
    warnings = []

    # === 0. Strip spaces in column names ===
    df.columns = df.columns.str.strip()

    # === 1. Check for missing values ===
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing.items():
            if count > 0:
                errors.append(f"❌ Column '{col}' has {count} missing values.")

    # === 2. Convert numeric columns safely ===
    numeric_cols = ['day', 'targeted_productivity', 'smv', 'wip', 'over_time',
                    'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 
                    'no_of_workers', 'actual_productivity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # convert, NaN if fails

            # Check for negative values
            if (df[col] < 0).any():
                errors.append(f"❌ Column '{col}' contains negative values.")

            # Special range check
            if col == 'actual_productivity':
                out_of_bounds = df[(df[col] < 0) | (df[col] > 1)]
                if not out_of_bounds.empty:
                    errors.append(f"❌ Column '{col}' has values outside [0,1] range.")

    # === 3. Check categorical columns ===
    categorical_cols = ['quarter', 'department', 'team']
    expected_values = {
        'quarter': ['Quarter1', 'Quarter2', 'Quarter3', 'Quarter4'],
        'department': ['sweing', 'finishing'],
        'team': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }

    for col in categorical_cols:
        if col in df.columns:
            if df[col].isnull().any():
                errors.append(f"❌ Column '{col}' contains missing values.")
            unexpected = set(df[col].dropna().unique()) - set(expected_values[col])
            if unexpected:
                warnings.append(f"⚠️ Column '{col}' contains unexpected values: {unexpected}")

    # === 4. Duplicate rows check ===
    if df.duplicated().any():
        warnings.append(f"⚠️ Dataset contains {df.duplicated().sum()} duplicate rows.")

    # === 5. Logical consistency checks ===
    if 'no_of_workers' in df.columns and (df['no_of_workers'] <= 0).any():
        errors.append("❌ Column 'no_of_workers' has values <= 0.")

    if 'smv' in df.columns and 'wip' in df.columns:
        inconsistent = df[(df['smv'] > 0) & ((df['wip'] <= 0) | (df['wip'].isnull()))]
        if not inconsistent.empty:
            warnings.append(f"⚠️ {len(inconsistent)} rows have smv>0 but wip<=0 or missing.")

    # === 6. Summary statistics for interpretation ===
    print("\n📊 Dataset Summary Statistics:")
    print(df.describe(include='all').transpose())

    print("\n🔍 Data Validation Report:")
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("✅ No critical errors found.")

    if warnings:
        for w in warnings:
            print(w)
    else:
        print("✅ No warnings found.")

if __name__ == "__main__":
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
        sys.exit(1)
    validate_data(df)

