# NYC 311 Closure Prediction

This project predicts whether a NYC 311 service request will be closed within 24 hours.

The final model is a `CatBoostClassifier` trained from `data/train.csv` and used to create predictions for `data/test.csv`.

## Files

```text
src/train_model.py          Final training and prediction script
config/model_columns.txt    Selected input columns
config/model_params.json    Tuned model parameters
data/train.csv              Training data
data/test.csv               Test data
data/submission.csv         Submission template
notebooks/                  Development and analysis notebooks
outputs/                    Generated results
```

## Method

The target variable is created from `Created Date` and `Closed Date`:

- `1`: the request closed within 24 hours
- `0`: the request did not close within 24 hours

`Closed Date` is not used as an input feature because it would leak the answer.

The script creates date/time, categorical, geographic, missing-value, and interaction features before training the model.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python3 src/train_model.py
```

This trains the model, validates it, runs 5-fold cross-validation, and writes the final output files.

For a faster run without cross-validation:

```bash
python3 src/train_model.py --no-cross-validate
```

## Results

Latest validation performance:

- Training accuracy: `0.9174`
- Validation accuracy: `0.9031`
- 5-fold mean validation accuracy: `0.9001`
- Train-validation gap: `0.0143`

## Outputs

Important generated files:

```text
outputs/model_submission.csv
outputs/model_summary.csv
outputs/classification_report.csv
outputs/confusion_matrix.csv
outputs/cross_validation_results.csv
outputs/generated_model_features.txt
outputs/selected_input_columns.txt
```

## Notes

- Keep `data/train.csv`, `data/test.csv`, and `data/submission.csv` unchanged.
- The test set has no labels, so accuracy is measured on a validation split from `data/train.csv`.
- The final model uses 16 selected input columns and creates 53 model features.
