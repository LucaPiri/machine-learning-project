# NYC 311 Closure Prediction

This project predicts whether a NYC 311 service request will be closed within 24 hours of being created.

The final model is a `CatBoostClassifier`. Raw data is first converted into processed feature files, then the model is trained on those files.

## Project Structure

```text
src/preprocessing.py       Cleaning and feature-engineering functions
src/preprocess_data.py     Builds processed train/test files
src/train_model.py         Trains the final model and creates predictions
config/                    Selected columns and model parameters
data/                      Raw train, test, and submission files
notebooks/                 Analysis and development notebooks
outputs/                   Generated processed data, reports, and submission
```

## Method

The target is `1` when a request closes within 24 hours and `0` otherwise. `Closed Date` is used only to create this target and is not used as a model feature.

## How To Run

Install dependencies:

```bash
pip install -r config/requirements.txt
```

Build processed data:

```bash
python3 src/preprocess_data.py
```

Train the model:

```bash
python3 src/train_model.py
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
outputs/processed/train_features.csv
outputs/processed/target.csv
outputs/processed/test_features.csv
outputs/processed/metadata.json
```
