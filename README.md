# NYC 311 Closure Prediction

This project predicts whether a NYC 311 service request will be closed within 24 hours of being created.

The final model is a `CatBoostClassifier`. Raw data is first converted into processed feature files, then the model is trained on those files.

## Project Structure

```text
config/preprocessing.py    Cleaning and feature-engineering functions
config/prepare_data.py     Builds processed train/test files
config/train_and_evaluate_model.py
                          Trains, validates, saves the model, and creates predictions
config/create_report_figures.py
                          Separately generates only the report figures
config/                    Model parameters, selected columns, and data-prep code
data/                      Raw train, test, and submission files
final_submission_notebook.ipynb
                          Notebook for reproducing final predictions
outputs/                   Generated when the scripts are run
```

## Method

The target is `1` when a request closes within 24 hours and `0` otherwise. `Closed Date` is used only to create this target and is not used as a model feature.

## How To Run

Install dependencies:

```bash
pip install -r config/requirements.txt
```

Simplest option: open and run every cell in:

```text
final_submission_notebook.ipynb
```

That notebook runs the whole workflow from raw CSV files to final outputs:
processed data, validation metrics, trained model, prediction CSV, and report
figures.

Equivalent command-line workflow:

Build processed data:

```bash
python3 config/prepare_data.py
```

Train the model and reproduce the cross-validation diagnostics:

```bash
python3 config/train_and_evaluate_model.py --cross-validate
```

Generate the figures used in the report. This is intentionally separate from
model training:

```bash
python3 config/create_report_figures.py
```

## Results

- Training accuracy: `0.9181`
- Validation accuracy: `0.9032`
- Train-validation gap: `0.0148`

## Outputs

```text
outputs/submission.csv
outputs/catboost_model.cbm
outputs/model_summary.csv
outputs/classification_report.csv
outputs/confusion_matrix.csv
outputs/cross_validation_results.csv
outputs/figures/
```

The preprocessing step also creates `outputs/processed/` as an intermediate
folder used by the training script. It can be regenerated at any time with
`python3 config/prepare_data.py`.
