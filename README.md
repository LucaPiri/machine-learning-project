# NYC 311 Closure Prediction

This project trains a machine learning model for NYC 311 service requests. The model predicts whether a request will be closed within 24 hours of its creation.

The final reproducible workflow lives in one Python script plus two configuration files. Exploratory notebooks are included only as supporting analysis for how the model was developed.

## Project Structure

```text
src/train_model.py          Main training, validation, and prediction script
config/model_columns.txt    Raw input columns used by the model
config/model_params.json    Tuned model hyperparameters
data/train.csv              Training data, including Closed Date for target creation
data/test.csv               Unlabeled test data used for final predictions
data/submission.csv         Submission template
notebooks/                  Exploratory analysis and model-development notebooks
requirements.txt            Python dependencies
```

Generated files are written to `outputs/` when the script runs. That folder is ignored by git and should not be committed.

## Notebook Workflow

The notebooks are ordered from `01` to `04`:

```text
notebooks/01_preliminary_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_feature_selection.ipynb
notebooks/04_logistic_regression.ipynb
```

They are useful for showing the exploration, preprocessing experiments, feature-selection reasoning, and baseline model comparison that led to the final script. The production result should still be reproduced from `src/train_model.py`.

## How The Model Works

The target is created from `Created Date` and `Closed Date`:

- `1` means the request closed within 24 hours.
- `0` means it did not close within 24 hours.

`Closed Date` is never used as an input feature because it is the answer source and is not available in `data/test.csv`.

The script then builds model features from the selected raw columns:

- date/time features from `Created Date`
- cleaned numeric and geographic fields
- categorical versions of location fields
- distance from NYC center
- categorical interaction features, such as agency/problem/borough combinations
- missing-value and text-length indicators

Categorical features are encoded with scikit-learn `TargetEncoder`, and the classifier is a regularized `HistGradientBoostingClassifier`.

## Run In VSCode

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the tuned model:

```bash
python3 src/train_model.py --columns-file config/model_columns.txt --params-file config/model_params.json
```

The latest tuned validation result is:

- Training accuracy: `0.9221`
- Validation accuracy: `0.9031`
- Train-validation gap: `0.0190`

The validation split is created from `data/train.csv` with a fixed random seed, so the result is reproducible.

## Outputs

Running the script creates:

```text
outputs/model_submission.csv
outputs/model_summary.csv
outputs/validation_results.csv
outputs/classification_report.csv
outputs/training_classification_report.csv
outputs/confusion_matrix.csv
outputs/generated_model_features.txt
outputs/selected_input_columns.txt
outputs/selected_model_params.json
outputs/final_model_artifact.joblib
```

These files are generated artifacts. Recreate them by running the script instead of committing them.

## Configuration

To change the raw input columns, edit:

```text
config/model_columns.txt
```

To change the model hyperparameters, edit:

```text
config/model_params.json
```

You can also override hyperparameters directly from the command line:

```bash
python3 src/train_model.py --max-iter 480 --learning-rate 0.035 --max-leaf-nodes 31 --min-samples-leaf 150 --l2-regularization 0.2
```

To list all raw columns available in `data/train.csv`:

```bash
python3 src/train_model.py --list-columns
```

## Notes

- Keep `data/train.csv`, `data/test.csv`, and `data/submission.csv` unchanged.
- Do not add `Closed Date` to `config/model_columns.txt`.
- `data/test.csv` has no labels, so reported accuracy and gap come from the validation split of `data/train.csv`.
- The current tuned model uses 38 raw input columns and creates 83 final model features.
- Commit source, config, and notebooks. Do not commit generated model outputs.
