import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder


DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
RANDOM_STATE = 42
MODEL_NAME = "HistGradientBoostingClassifier with target-encoded features"
DEFAULT_INPUT_COLUMNS = [
    "Unique Key",
    "Created Date",
    "Agency",
    "Agency Name",
    "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)",
    "Additional Details",
    "Location Type",
    "Incident Zip",
    "Incident Address",
    "Street Name",
    "Cross Street 1",
    "Cross Street 2",
    "Intersection Street 1",
    "Intersection Street 2",
    "Address Type",
    "City",
    "Landmark",
    "Facility Type",
    "Community Board",
    "Council District",
    "Police Precinct",
    "BBL",
    "Borough",
    "X Coordinate (State Plane)",
    "Y Coordinate (State Plane)",
    "Open Data Channel Type",
    "Park Facility Name",
    "Park Borough",
    "Vehicle Type",
    "Taxi Company Borough",
    "Taxi Pick Up Location",
    "Bridge Highway Name",
    "Bridge Highway Direction",
    "Road Ramp",
    "Bridge Highway Segment",
    "Latitude",
    "Longitude",
]
FORBIDDEN_FEATURE_COLUMNS = {"Closed Date"}
DEFAULT_MODEL_PARAMS = {
    "max_iter": 480,
    "learning_rate": 0.035,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 150,
    "l2_regularization": 0.2,
}
MODEL_PARAM_TYPES = {
    "max_iter": int,
    "learning_rate": float,
    "max_leaf_nodes": int,
    "min_samples_leaf": int,
    "l2_regularization": float,
}


def parse_column_list(columns_text):
    if columns_text is None:
        return DEFAULT_INPUT_COLUMNS

    return [
        col.strip()
        for col in columns_text.split(",")
        if col.strip()
    ]


def load_column_list(columns_file):
    columns_path = Path(columns_file)
    return [
        line.strip()
        for line in columns_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_model_params(params_file):
    params_path = Path(params_file)
    loaded_params = json.loads(params_path.read_text())

    if not isinstance(loaded_params, dict):
        raise ValueError("Model parameter file must contain a JSON object.")

    unknown_params = sorted(set(loaded_params) - set(DEFAULT_MODEL_PARAMS))
    if unknown_params:
        raise ValueError(f"Unknown model parameters: {unknown_params}")

    params = DEFAULT_MODEL_PARAMS.copy()
    for name, value in loaded_params.items():
        params[name] = MODEL_PARAM_TYPES[name](value)

    return params


def parse_model_params(args):
    if args.params_file:
        params = load_model_params(args.params_file)
    else:
        params = DEFAULT_MODEL_PARAMS.copy()

    cli_overrides = {
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2_regularization,
    }
    for name, value in cli_overrides.items():
        if value is not None:
            params[name] = MODEL_PARAM_TYPES[name](value)

    return params


def validate_input_columns(columns, train_df, test_df):
    if not columns:
        raise ValueError("Select at least one input column.")

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


def make_target(train_df):
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


def add_created_date_features(df):
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

    df["created_hour_sin"] = np.sin(2 * np.pi * df["created_hour"] / 24)
    df["created_hour_cos"] = np.cos(2 * np.pi * df["created_hour"] / 24)
    df["created_day_sin"] = np.sin(2 * np.pi * df["created_day_of_week"] / 7)
    df["created_day_cos"] = np.cos(2 * np.pi * df["created_day_of_week"] / 7)

    return df


def add_numeric_and_geo_features(df):
    df = df.copy()

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
            df[f"{col}_cat"] = (
                df[col]
                .round(0)
                .astype("Int64")
                .astype(str)
                .replace("<NA>", "UNKNOWN")
            )

    if {"Latitude", "Longitude"}.issubset(df.columns):
        nyc_latitude = 40.7128
        nyc_longitude = -74.0060
        df["distance_from_nyc_center"] = np.sqrt(
            (df["Latitude"] - nyc_latitude) ** 2
            + (df["Longitude"] - nyc_longitude) ** 2
        )

    return df


def add_categorical_interactions(df):
    df = df.copy()

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
                df[left_col].astype(str).fillna("UNKNOWN")
                + " | "
                + df[right_col].astype(str).fillna("UNKNOWN")
            )

    return df


def add_high_signal_interactions(df):
    df = df.copy()

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
            interaction = df[cols[0]].astype(str).fillna("UNKNOWN")
            for col in cols[1:]:
                interaction = (
                    interaction
                    + " | "
                    + df[col].astype(str).fillna("UNKNOWN")
                )
            df["__".join(cols)] = interaction

    return df


def add_missing_and_length_features(df):
    df = df.copy()

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
    df = df[input_columns].copy()

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

    return df.drop(columns=leakage_or_raw_cols, errors="ignore")


def build_model(feature_df, model_params=None):
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    categorical_cols = feature_df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_cols = [
        col for col in feature_df.columns
        if col not in categorical_cols
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "target_encoder",
                TargetEncoder(
                    target_type="binary",
                    cv=5,
                    smooth="auto",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=model_params["max_iter"],
        learning_rate=model_params["learning_rate"],
        max_leaf_nodes=model_params["max_leaf_nodes"],
        min_samples_leaf=model_params["min_samples_leaf"],
        l2_regularization=model_params["l2_regularization"],
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_outputs(
    outputs_dir,
    model,
    submission,
    test_predictions,
    validation_predictions,
    training_predictions,
    y_train,
    y_valid,
    training_accuracy,
    validation_accuracy,
    feature_count,
    input_columns,
    output_features,
    model_params,
):
    outputs_dir.mkdir(exist_ok=True)

    prediction_col = "prediction"
    if prediction_col not in submission.columns:
        prediction_col = submission.columns[-1]

    model_submission = submission.copy()
    model_submission[prediction_col] = test_predictions.astype(int)
    model_submission.to_csv(outputs_dir / "model_submission.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "training_accuracy": round(training_accuracy, 4),
                "validation_accuracy": round(validation_accuracy, 4),
                "train_validation_gap": round(training_accuracy - validation_accuracy, 4),
                "input_columns": len(input_columns),
                "selected_features": feature_count,
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        ]
    )
    summary.to_csv(outputs_dir / "model_summary.csv", index=False)
    summary.rename(columns={"model": "final_model"}).to_csv(
        outputs_dir / "validation_results.csv",
        index=False,
    )

    report = pd.DataFrame(
        classification_report(
            y_valid,
            validation_predictions,
            output_dict=True,
        )
    ).transpose()
    report.to_csv(outputs_dir / "classification_report.csv")

    training_report = pd.DataFrame(
        classification_report(
            y_train,
            training_predictions,
            output_dict=True,
        )
    ).transpose()
    training_report.to_csv(outputs_dir / "training_classification_report.csv")

    confusion = pd.DataFrame(
        confusion_matrix(y_valid, validation_predictions),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    confusion.to_csv(outputs_dir / "confusion_matrix.csv")

    joblib.dump(model, outputs_dir / "final_model_artifact.joblib")
    (outputs_dir / "selected_input_columns.txt").write_text(
        "\n".join(input_columns) + "\n"
    )
    (outputs_dir / "generated_model_features.txt").write_text(
        "\n".join(output_features) + "\n"
    )
    (outputs_dir / "selected_model_params.json").write_text(
        json.dumps(model_params, indent=2, sort_keys=True) + "\n"
    )

    return summary, report, confusion


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the NYC 311 closure-time model with a selectable raw "
            "input-column list."
        )
    )
    parser.add_argument(
        "--columns",
        help=(
            "Comma-separated raw columns to use. Example: "
            "'Created Date,Agency,Borough,Incident Zip'"
        ),
    )
    parser.add_argument(
        "--columns-file",
        help=(
            "Text file containing one raw column name per line. Lines starting "
            "with # are ignored."
        ),
    )
    parser.add_argument(
        "--params-file",
        help=(
            "JSON file containing model hyperparameters. Supported keys: "
            f"{', '.join(DEFAULT_MODEL_PARAMS)}."
        ),
    )
    parser.add_argument("--max-iter", type=int, help="Boosting iterations.")
    parser.add_argument("--learning-rate", type=float, help="Boosting learning rate.")
    parser.add_argument("--max-leaf-nodes", type=int, help="Maximum leaves per tree.")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print the raw columns available in train.csv and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "submission.csv")

    if args.list_columns:
        for col in train.columns:
            print(col)
        return

    if args.columns and args.columns_file:
        raise ValueError("Use either --columns or --columns-file, not both.")

    if args.columns_file:
        input_columns = load_column_list(args.columns_file)
    else:
        input_columns = parse_column_list(args.columns)

    input_columns = validate_input_columns(input_columns, train, test)
    model_params = parse_model_params(args)

    y = make_target(train)
    X = make_features(train, input_columns)
    X_test = make_features(test, input_columns).reindex(
        columns=X.columns,
        fill_value=np.nan,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_model = build_model(X, model_params)
    validation_preprocessor = validation_model.named_steps["preprocessor"]
    validation_classifier = validation_model.named_steps["classifier"]
    X_train_prepared = validation_preprocessor.fit_transform(X_train, y_train)
    validation_classifier.fit(X_train_prepared, y_train)
    training_predictions = validation_classifier.predict(
        X_train_prepared
    ).astype(int)
    validation_predictions = validation_model.predict(X_valid).astype(int)
    training_accuracy = accuracy_score(y_train, training_predictions)
    validation_accuracy = accuracy_score(y_valid, validation_predictions)

    final_model = build_model(X, model_params)
    final_model.fit(X, y)
    test_predictions = final_model.predict(X_test).astype(int)

    summary, report, confusion = save_outputs(
        outputs_dir=outputs_dir,
        model=final_model,
        submission=submission,
        test_predictions=test_predictions,
        validation_predictions=validation_predictions,
        training_predictions=training_predictions,
        y_train=y_train,
        y_valid=y_valid,
        training_accuracy=training_accuracy,
        validation_accuracy=validation_accuracy,
        feature_count=X.shape[1],
        input_columns=input_columns,
        output_features=X.columns.tolist(),
        model_params=model_params,
    )

    print(f"Final model: {MODEL_NAME}")
    print(f"Training accuracy: {training_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Train-validation gap: {training_accuracy - validation_accuracy:.4f}")
    print(f"Input columns: {len(input_columns)}")
    print(f"Feature count: {X.shape[1]}")
    print("Model parameters:")
    for name, value in model_params.items():
        print(f"- {name}: {value}")
    print("Raw columns used:")
    for col in input_columns:
        print(f"- {col}")
    print(f"Saved outputs to: {outputs_dir}")
    print()
    print(summary.to_string(index=False))
    print()
    print(report.round(4).to_string())
    print()
    print(confusion.to_string())


if __name__ == "__main__":
    main()
