"""Train and validate the NYC 311 24-hour closure model.

Run src/preprocess_data.py before this script so the processed feature files are
available in outputs/processed/.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocessing import DEFAULT_PROCESSED_DIR, load_processed_data


RANDOM_STATE = 42
MODEL_NAME = "CatBoostClassifier with native categorical features"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_FILE = PROJECT_ROOT / "config" / "model_params.json"
DEFAULT_MODEL_PARAMS = {
    "iterations": 350,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
}
MODEL_PARAM_TYPES = {
    "iterations": int,
    "learning_rate": float,
    "depth": int,
    "l2_leaf_reg": float,
    "random_strength": float,
}


def load_model_params(params_file):
    """Load CatBoost hyperparameters from JSON and validate the allowed keys."""
    params_path = Path(params_file)
    loaded_params = json.loads(params_path.read_text())

    if not isinstance(loaded_params, dict):
        raise ValueError("Model parameter file must contain a JSON object.")

    # Fail fast on misspelled parameter names.
    unknown_params = sorted(set(loaded_params) - set(DEFAULT_MODEL_PARAMS))
    if unknown_params:
        raise ValueError(f"Unknown model parameters: {unknown_params}")

    params = DEFAULT_MODEL_PARAMS.copy()
    for name, value in loaded_params.items():
        params[name] = MODEL_PARAM_TYPES[name](value)

    return params


def parse_model_params(args):
    """Combine default, file-based, and command-line model parameters."""
    if args.params_file:
        params = load_model_params(args.params_file)
    elif DEFAULT_PARAMS_FILE.exists():
        params = load_model_params(DEFAULT_PARAMS_FILE)
    else:
        params = DEFAULT_MODEL_PARAMS.copy()

    # Command-line values override the config for quick experiments.
    cli_overrides = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
    }
    for name, value in cli_overrides.items():
        if value is not None:
            params[name] = MODEL_PARAM_TYPES[name](value)

    return params


def build_model(model_params):
    """Construct a CatBoost classifier using the configured hyperparameters."""
    # CatBoost can use the many service-request categories directly.
    return CatBoostClassifier(
        iterations=model_params["iterations"],
        learning_rate=model_params["learning_rate"],
        depth=model_params["depth"],
        l2_leaf_reg=model_params["l2_leaf_reg"],
        random_strength=model_params["random_strength"],
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def evaluate_train_valid_split(X, y, categorical_cols, model_params):
    """Fit one validation model and return holdout predictions and accuracy scores."""
    # Keep the validation score reproducible.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_model = build_model(model_params)
    # Keep the best iteration from the validation set.
    validation_model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    training_predictions = validation_model.predict(X_train).astype(int)
    validation_predictions = validation_model.predict(X_valid).astype(int)

    return {
        "validation_predictions": validation_predictions,
        "y_valid": y_valid,
        "training_accuracy": accuracy_score(y_train, training_predictions),
        "validation_accuracy": accuracy_score(y_valid, validation_predictions),
    }


def cross_validate_model(X, y, categorical_cols, model_params, n_splits=5):
    """Run stratified cross-validation to measure score stability across folds."""
    rows = []
    # Keep the 0/1 class balance similar in each fold.
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = build_model(model_params)
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_cols,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )

        training_predictions = model.predict(X_train).astype(int)
        validation_predictions = model.predict(X_valid).astype(int)
        training_accuracy = accuracy_score(y_train, training_predictions)
        validation_accuracy = accuracy_score(y_valid, validation_predictions)

        rows.append({
            "fold": fold,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "training_error": 1 - training_accuracy,
            "validation_error": 1 - validation_accuracy,
            "train_validation_gap": training_accuracy - validation_accuracy,
        })

    results = pd.DataFrame(rows)
    # A large train-vs-validation gap would point to overfitting.
    summary = {
        "folds": n_splits,
        "mean_training_accuracy": results["training_accuracy"].mean(),
        "mean_validation_accuracy": results["validation_accuracy"].mean(),
        "mean_training_error": results["training_error"].mean(),
        "mean_validation_error": results["validation_error"].mean(),
        "mean_gap": results["train_validation_gap"].mean(),
        "max_gap": results["train_validation_gap"].max(),
        "min_gap": results["train_validation_gap"].min(),
        "std_validation_accuracy_sample": results["validation_accuracy"].std(ddof=1),
        "variance_validation_accuracy_sample": results["validation_accuracy"].var(ddof=1),
        "min_validation_accuracy": results["validation_accuracy"].min(),
        "max_validation_accuracy": results["validation_accuracy"].max(),
        "validation_accuracy_range": (
            results["validation_accuracy"].max()
            - results["validation_accuracy"].min()
        ),
    }
    return results, summary


def save_outputs(
    outputs_dir,
    submission,
    test_predictions,
    validation_predictions,
    y_valid,
    training_accuracy,
    validation_accuracy,
    cv_summary,
    metadata,
):
    """Save submission, validation diagnostics, and feature lists to outputs/."""
    outputs_dir.mkdir(exist_ok=True)

    # Preserve the submission template and replace only the prediction column.
    prediction_col = "prediction"
    if prediction_col not in submission.columns:
        prediction_col = submission.columns[-1]

    model_submission = submission.copy()
    model_submission[prediction_col] = test_predictions.astype(int)
    model_submission.to_csv(outputs_dir / "model_submission.csv", index=False)

    summary = pd.DataFrame([{
        "model": MODEL_NAME,
        "training_accuracy": round(training_accuracy, 4),
        "validation_accuracy": round(validation_accuracy, 4),
        "training_error": round(1 - training_accuracy, 4),
        "validation_error": round(1 - validation_accuracy, 4),
        "train_validation_gap": round(training_accuracy - validation_accuracy, 4),
        "cv_mean_validation_accuracy": (
            round(cv_summary["mean_validation_accuracy"], 4)
            if cv_summary else None
        ),
        "cv_validation_variance": (
            round(cv_summary["variance_validation_accuracy_sample"], 8)
            if cv_summary else None
        ),
        "cv_validation_std": (
            round(cv_summary["std_validation_accuracy_sample"], 4)
            if cv_summary else None
        ),
        "cv_mean_gap": round(cv_summary["mean_gap"], 4) if cv_summary else None,
        "input_columns": len(metadata["input_columns"]),
        "selected_features": len(metadata["output_features"]),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }])
    summary.to_csv(outputs_dir / "model_summary.csv", index=False)

    report = pd.DataFrame(
        classification_report(
            y_valid,
            validation_predictions,
            output_dict=True,
        )
    ).transpose()
    report.to_csv(outputs_dir / "classification_report.csv")

    confusion = pd.DataFrame(
        confusion_matrix(y_valid, validation_predictions),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    confusion.to_csv(outputs_dir / "confusion_matrix.csv")

    (outputs_dir / "selected_input_columns.txt").write_text(
        "\n".join(metadata["input_columns"]) + "\n"
    )
    (outputs_dir / "generated_model_features.txt").write_text(
        "\n".join(metadata["output_features"]) + "\n"
    )

    return summary, report, confusion


def parse_args():
    """Define command-line options for training from processed data."""
    parser = argparse.ArgumentParser(
        description="Train the NYC 311 closure-time model from processed files."
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing processed files from src/preprocess_data.py.",
    )
    parser.add_argument(
        "--params-file",
        help=(
            "JSON file containing model hyperparameters. Supported keys: "
            f"{', '.join(DEFAULT_MODEL_PARAMS)}."
        ),
    )
    parser.add_argument("--iterations", type=int, help="Boosting iterations.")
    parser.add_argument("--learning-rate", type=float, help="Boosting learning rate.")
    parser.add_argument("--depth", type=int, help="CatBoost tree depth.")
    parser.add_argument("--l2-leaf-reg", type=float, help="CatBoost L2 regularization.")
    parser.add_argument(
        "--random-strength",
        type=float,
        help="CatBoost random score strength.",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        default=False,
        help="Run 5-fold cross-validation diagnostics. This is slower.",
    )
    return parser.parse_args()


def main():
    """Load processed features, validate the model, and save predictions."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    X_train, y_train, X_test, metadata = load_processed_data(args.processed_dir)
    categorical_cols = metadata["categorical_cols"]
    model_params = parse_model_params(args)

    if len(X_test) != len(pd.read_csv(data_dir / "submission.csv")):
        raise ValueError("Processed test rows do not match submission template rows.")

    split_result = evaluate_train_valid_split(
        X_train,
        y_train,
        categorical_cols,
        model_params,
    )
    validation_predictions = split_result["validation_predictions"]
    y_valid = split_result["y_valid"]
    training_accuracy = split_result["training_accuracy"]
    validation_accuracy = split_result["validation_accuracy"]

    cv_summary = None
    if args.cross_validate:
        cv_results, cv_summary = cross_validate_model(
            X_train,
            y_train,
            categorical_cols,
            model_params,
        )
        outputs_dir.mkdir(exist_ok=True)
        cv_results.to_csv(outputs_dir / "cross_validation_results.csv", index=False)

    # Train once more on all labeled rows before predicting the test set.
    final_model = build_model(model_params)
    final_model.fit(X_train, y_train, cat_features=categorical_cols)
    test_predictions = final_model.predict(X_test).astype(int)

    submission = pd.read_csv(data_dir / "submission.csv")
    summary, report, confusion = save_outputs(
        outputs_dir=outputs_dir,
        submission=submission,
        test_predictions=test_predictions,
        validation_predictions=validation_predictions,
        y_valid=y_valid,
        training_accuracy=training_accuracy,
        validation_accuracy=validation_accuracy,
        cv_summary=cv_summary,
        metadata=metadata,
    )

    print(f"Final model: {MODEL_NAME}")
    print(f"Training accuracy: {training_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Training error (bias proxy): {1 - training_accuracy:.4f}")
    print(f"Validation error: {1 - validation_accuracy:.4f}")
    print(f"Train-validation gap: {training_accuracy - validation_accuracy:.4f}")
    if cv_summary:
        print("5-fold bias/variance diagnostics:")
        print(f"- Mean training accuracy: {cv_summary['mean_training_accuracy']:.4f}")
        print(f"- Mean validation accuracy: {cv_summary['mean_validation_accuracy']:.4f}")
        print(f"- Mean gap: {cv_summary['mean_gap']:.4f}")
        print(f"- Validation std: {cv_summary['std_validation_accuracy_sample']:.4f}")
    print(f"Input columns: {len(metadata['input_columns'])}")
    print(f"Feature count: {len(metadata['output_features'])}")
    print(f"Saved outputs to: {outputs_dir}")
    print()
    print(summary.to_string(index=False))
    print()
    print(report.round(4).to_string())
    print()
    print(confusion.to_string())


if __name__ == "__main__":
    main()
