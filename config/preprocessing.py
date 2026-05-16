"""Data cleaning and feature engineering for the NYC 311 model.

The functions in this file start from the raw train/test CSV files and return
CatBoost-ready feature matrices in memory. They deliberately do not write
intermediate feature CSVs, so a clean run only produces the final outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS_FILE = PROJECT_ROOT / "config" / "model_columns.txt"
UNKNOWN_CATEGORY = "UNKNOWN"

# Normal runs read config/model_columns.txt. This fallback just keeps the script
# usable if the config file is moved or missing.
FALLBACK_INPUT_COLUMNS = [
    "Created Date",
    "Agency",
    "Agency Name",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Additional Details",
    "Location Type",
    "Incident Zip",
    "Address Type",
    "City",
    "Community Board",
    "Police Precinct",
    "Borough",
    "Open Data Channel Type",
    "Latitude",
    "Longitude",
]
FORBIDDEN_FEATURE_COLUMNS = {"Closed Date"}


def load_column_list(columns_file):
    """Read a text file containing one raw input column name per line."""
    columns_path = Path(columns_file)
    # Allow blank lines and comments to keep the config file readable.
    return [
        line.strip()
        for line in columns_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_column_list(columns_text):
    """Return the raw input columns requested by the user or the project default."""
    # Prefer the checked-in column list unless the user passes an override.
    if columns_text is None:
        if DEFAULT_COLUMNS_FILE.exists():
            return load_column_list(DEFAULT_COLUMNS_FILE)
        return FALLBACK_INPUT_COLUMNS

    # Comma-separated overrides are useful for quick feature experiments.
    return [
        col.strip()
        for col in columns_text.split(",")
        if col.strip()
    ]


def validate_input_columns(columns, train_df, test_df):
    """Check that selected feature columns are usable for both train and test."""
    if not columns:
        raise ValueError("Select at least one input column.")

    # Every input feature must exist in both train and test.
    duplicate_columns = sorted({
        col for col in columns
        if columns.count(col) > 1
    })
    if duplicate_columns:
        raise ValueError(f"Duplicate selected columns: {duplicate_columns}")

    forbidden_columns = sorted(set(columns) & FORBIDDEN_FEATURE_COLUMNS)
    if forbidden_columns:
        raise ValueError(
            "These columns cannot be used as model inputs because they leak "
            f"the answer: {forbidden_columns}"
        )

    missing_train = sorted(set(columns) - set(train_df.columns))
    missing_test = sorted(set(columns) - set(test_df.columns))

    if missing_train:
        raise ValueError(f"Columns missing from train.csv: {missing_train}")

    if missing_test:
        raise ValueError(f"Columns missing from test.csv: {missing_test}")

    return columns


def dedupe_preserving_order(columns):
    """Remove duplicate column names without changing the original order."""
    return list(dict.fromkeys(columns))


def make_target(train_df):
    """Create the binary label: 1 when a request closes within 24 hours."""
    # Closed Date is used only here to build the label, never as a model input.
    created = pd.to_datetime(
        train_df["Created Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    closed = pd.to_datetime(
        train_df["Closed Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    hours_to_close = (closed - created).dt.total_seconds() / 3600
    return ((hours_to_close >= 0) & (hours_to_close <= 24)).astype(int)


def clean_numeric_column(series, decimal_comma=False):
    """Convert messy numeric text into numeric values while preserving NaNs."""
    # Coordinates may use decimal commas; ID-like fields may use thousands commas.
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    if decimal_comma:
        cleaned = cleaned.str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def clean_categorical_text(series):
    """Return a string series where missing categorical values are explicit."""
    return series.astype("string").fillna(UNKNOWN_CATEGORY).astype(str)


def add_created_date_features(df):
    """Expand Created Date into hour, weekday, weekend, and cyclic time features."""
    df = df.copy()
    created = pd.to_datetime(
        df["Created Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )

    df["created_hour"] = created.dt.hour
    df["created_day_of_week"] = created.dt.dayofweek
    df["created_day"] = created.dt.day
    df["created_is_weekend"] = created.dt.dayofweek.isin([5, 6]).astype(int)
    df["created_is_business_hour"] = created.dt.hour.between(9, 17).astype(int)
    df["created_minute_of_day"] = created.dt.hour * 60 + created.dt.minute

    # Keep midnight-adjacent hours close to each other numerically.
    df["created_hour_sin"] = np.sin(2 * np.pi * df["created_hour"] / 24)
    df["created_hour_cos"] = np.cos(2 * np.pi * df["created_hour"] / 24)
    df["created_day_sin"] = np.sin(2 * np.pi * df["created_day_of_week"] / 7)
    df["created_day_cos"] = np.cos(2 * np.pi * df["created_day_of_week"] / 7)

    return df


def add_numeric_and_geo_features(df):
    """Normalize numeric columns and add simple geographic helper features."""
    df = df.copy()

    # These sometimes arrive as formatted strings rather than clean numbers.
    numeric_cols = [
        "Unique Key",
        "Incident Zip",
        "Council District",
        "BBL",
        "X Coordinate (State Plane)",
        "Y Coordinate (State Plane)",
        "Latitude",
        "Longitude",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        df[col] = clean_numeric_column(
            df[col],
            decimal_comma=col in {"Latitude", "Longitude"},
        )

        if col != "Unique Key":
            # Keep a coarse categorical version of location/code-like numbers.
            df[f"{col}_cat"] = clean_categorical_text(
                df[col].round(0).astype("Int64")
            )

    if {"Latitude", "Longitude"}.issubset(df.columns):
        nyc_latitude = 40.7128
        nyc_longitude = -74.0060
        # A simple location proxy; no exact distance calculation is needed here.
        df["distance_from_nyc_center"] = np.sqrt(
            (df["Latitude"] - nyc_latitude) ** 2
            + (df["Longitude"] - nyc_longitude) ** 2
        )

    return df


def add_categorical_interactions(df):
    """Create pairwise categorical combinations that CatBoost can learn from."""
    df = df.copy()

    # Common service patterns often depend on a pair of fields, not just one.
    pairs = [
        ("Agency", "Problem (formerly Complaint Type)"),
        ("Agency", "Problem Detail (formerly Descriptor)"),
        (
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
        ),
        ("Agency", "Borough"),
        ("Problem (formerly Complaint Type)", "Borough"),
        ("Problem Detail (formerly Descriptor)", "Borough"),
        ("Incident Zip_cat", "Problem (formerly Complaint Type)"),
        ("Incident Zip_cat", "Problem Detail (formerly Descriptor)"),
        ("Open Data Channel Type", "Problem (formerly Complaint Type)"),
        ("Location Type", "Problem (formerly Complaint Type)"),
        ("Community Board", "Problem (formerly Complaint Type)"),
        ("Police Precinct", "Problem (formerly Complaint Type)"),
    ]

    for left_col, right_col in pairs:
        if left_col in df.columns and right_col in df.columns:
            df[f"{left_col}__{right_col}"] = (
                clean_categorical_text(df[left_col])
                + " | "
                + clean_categorical_text(df[right_col])
            )

    return df


def add_high_signal_interactions(df):
    """Create larger categorical combinations found useful during exploration."""
    df = df.copy()

    # These larger combinations performed well during the feature-selection work.
    groups = [
        (
            "Agency",
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
        ),
        (
            "Agency",
            "Problem (formerly Complaint Type)",
            "Borough",
        ),
        (
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
            "Borough",
        ),
        (
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
            "Location Type",
        ),
        (
            "Incident Zip_cat",
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
        ),
        (
            "Community Board",
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
        ),
        (
            "Open Data Channel Type",
            "Agency",
            "Problem (formerly Complaint Type)",
        ),
        (
            "Agency",
            "Problem (formerly Complaint Type)",
            "Problem Detail (formerly Descriptor)",
            "Borough",
        ),
    ]

    for cols in groups:
        if all(col in df.columns for col in cols):
            # CatBoost can use these string combinations directly as categories.
            interaction = clean_categorical_text(df[cols[0]])
            for col in cols[1:]:
                interaction = (
                    interaction
                    + " | "
                    + clean_categorical_text(df[col])
                )
            df["__".join(cols)] = interaction

    return df


def add_missing_and_length_features(df):
    """Add flags and lengths for text fields where missingness may be predictive."""
    df = df.copy()

    # Missing text and description length can both carry useful signal.
    text_cols = [
        "Problem Detail (formerly Descriptor)",
        "Additional Details",
        "Incident Address",
        "Street Name",
    ]

    for col in text_cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
            df[f"{col}_length"] = df[col].astype(str).str.len().where(
                ~df[col].isna(),
                0,
            )

    return df


def make_features(df, input_columns):
    """Build the full model feature table from the selected raw columns."""
    df = df[input_columns].copy()

    # Keep train and test on the exact same feature pipeline.
    if "Created Date" in df.columns:
        df = add_created_date_features(df)

    df = add_numeric_and_geo_features(df)
    df = add_categorical_interactions(df)
    df = add_high_signal_interactions(df)
    df = add_missing_and_length_features(df)

    leakage_or_raw_cols = [
        "Created Date",
        "Location",
        "Unnamed: 0",
    ]

    # Drop raw helper columns after extracting the useful parts.
    return df.drop(columns=leakage_or_raw_cols, errors="ignore")


def prepare_catboost_features(feature_df, categorical_cols=None):
    """Convert feature columns into CatBoost-friendly numeric and string dtypes."""
    prepared = feature_df.copy()
    if categorical_cols is None:
        categorical_cols = prepared.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
    else:
        # Use the train schema so test columns are treated the same way.
        categorical_cols = list(categorical_cols)
        missing_categorical_cols = sorted(set(categorical_cols) - set(prepared.columns))
        if missing_categorical_cols:
            raise ValueError(
                "Categorical columns missing from feature dataframe: "
                f"{missing_categorical_cols}"
            )

    numeric_cols = [
        col for col in prepared.columns
        if col not in categorical_cols
    ]

    for col in categorical_cols:
        # CatBoost accepts categorical columns, but missing markers should be explicit.
        prepared[col] = clean_categorical_text(prepared[col])

    for col in numeric_cols:
        # Leave invalid numbers as NaN; CatBoost can handle them.
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype(float)

    return prepared, categorical_cols


def build_processed_datasets(data_dir, input_columns):
    """Load raw files and return CatBoost-ready train/test matrices in memory."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_header = pd.read_csv(train_path, nrows=0)
    test_header = pd.read_csv(test_path, nrows=0)
    input_columns = validate_input_columns(input_columns, train_header, test_header)

    target_columns = ["Created Date", "Closed Date"]
    missing_target_columns = sorted(set(target_columns) - set(train_header.columns))
    if missing_target_columns:
        raise ValueError(
            "Columns missing from train.csv for target creation: "
            f"{missing_target_columns}"
        )

    train = pd.read_csv(
        train_path,
        usecols=dedupe_preserving_order(input_columns + target_columns),
    )
    test = pd.read_csv(test_path, usecols=input_columns)

    y_train = make_target(train)
    X_train = make_features(train, input_columns)
    X_test = make_features(test, input_columns).reindex(
        columns=X_train.columns,
        fill_value=np.nan,
    )

    X_train_prepared, categorical_cols = prepare_catboost_features(X_train)
    X_test_prepared, _ = prepare_catboost_features(
        X_test,
        categorical_cols=categorical_cols,
    )

    metadata = {
        "input_columns": input_columns,
        "output_features": X_train_prepared.columns.tolist(),
        "categorical_cols": categorical_cols,
        "train_rows": int(X_train_prepared.shape[0]),
        "test_rows": int(X_test_prepared.shape[0]),
    }

    return X_train_prepared, y_train, X_test_prepared, metadata
