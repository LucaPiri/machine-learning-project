# Machine Learning Project
This repository contains the code and notebooks for a machine learning project based on NYC 311 Service Requests.
The goal of the project is to build a model that predicts whether a service request will be closed within 24 hours of its creation.

# Project document
The project outline is available here:
[Project outline](https://docs.google.com/document/d/14a6ASEwYOI9S9bXGaAILKr5AFJ11suaY5iGhtz6E9DQ/edit?usp=sharing)

# Repository workflow
Before working on the project, always update your local repository to avoid conflicts with other team members:
git pull
git status

# Notebook workflow
The cleaned notebook flow is:

1. `notebooks/01_preliminary_eda.ipynb` - exploratory analysis only.
2. `notebooks/03_preprocessing.ipynb` - creates `X_train`, `y_train`, and `X_test`.
3. `notebooks/04_feature_selection.ipynb` - selects the modeling feature set.
4. `notebooks/06_logistic_regression.ipynb` - compares baseline and gradient-boosting models, then prepares test predictions.

# Main libraries
The project uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `optuna`.
For stronger tabular models, the modeling notebook also supports `LightGBM`, `XGBoost`, and `CatBoost` when they are installed.

# Commit
After making changes commit and push them:
git add .
git commit -m "Write a clear commit message"
git push

Do not modify the original dataset files directly. Work inside notebooks or scripts and keep the raw CSV files unchanged.

Make sure to only commit the code of the Notebooks and not also the output, as to prevent useless code commit
