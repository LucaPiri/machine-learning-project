#Create processed train/test feature files from the raw NYC 311 CSVs.

import argparse
from pathlib import Path

import pandas as pd

from preprocessing import (
    DEFAULT_PROCESSED_DIR,
    build_processed_datasets,
    load_column_list,
    parse_column_list,
    save_processed_data,
)


def parse_args():
    """Parse options for building the processed feature files."""
    parser = argparse.ArgumentParser(
        description="Create processed feature files for model training."
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
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory where processed train/test files will be saved.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print the raw columns available in train.csv and exit.",
    )
    return parser.parse_args()


def main():
    """Load raw data, create model features, and save processed CSV files."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    train_header = pd.read_csv(data_dir / "train.csv", nrows=0)
    if args.list_columns:
        # Handy when checking column names before editing model_columns.txt.
        for col in train_header.columns:
            print(col)
        return

    if args.columns and args.columns_file:
        raise ValueError("Use either --columns or --columns-file, not both.")

    if args.columns_file:
        # In the normal project run this is config/model_columns.txt.
        input_columns = load_column_list(args.columns_file)
    else:
        input_columns = parse_column_list(args.columns)

    X_train, y_train, X_test, metadata = build_processed_datasets(
        data_dir=data_dir,
        input_columns=input_columns,
    )
    paths = save_processed_data(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metadata=metadata,
        processed_dir=args.processed_dir,
    )

    print("Processed data saved:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    print(f"Training rows: {metadata['train_rows']}")
    print(f"Test rows: {metadata['test_rows']}")
    print(f"Features: {len(metadata['output_features'])}")
    print(f"Categorical features: {len(metadata['categorical_cols'])}")


if __name__ == "__main__":
    main()
