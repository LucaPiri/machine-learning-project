# NYC 311 Closure Prediction

This project predicts whether a NYC 311 service request will be closed within 24 hours of being created.

The final model is a `CatBoostClassifier`. Raw data is cleaned and transformed in memory, then the model is trained on the engineered features.

## Project Structure

```text
config/model_columns.txt   Raw columns selected for the final model
config/model_params.json   CatBoost hyperparameters used in the final run
config/preprocessing.py    Cleaning and feature-engineering functions
config/train_and_evaluate_model.py
                          Trains, validates, and creates predictions
config/create_report_figures.py
                          Separately generates only the report figures
data/                      Raw train, test, and submission-template files
final_submission_notebook.ipynb
                          One-click notebook for reproducing the workflow
outputs/                   Final generated files after running the project
```

## Method

The target is `1` when a request closes within 24 hours and `0` otherwise.
`Closed Date` is used only to create this target and is never used as a model
feature. This avoids target leakage.

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
validation metrics, prediction CSV, and report figures. It does not write
intermediate processed datasets or a saved model artifact.

Equivalent command-line workflow:

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
outputs/model_summary.csv
outputs/classification_report.csv
outputs/confusion_matrix.csv
outputs/cross_validation_results.csv
outputs/figures/
```
