# B2B Sales Forecasting Ensemble (Capstone)

Capstone repository for forecasting quarterly company sales using supervised machine learning, feature engineering, and model optimization.

## Course Information

- Course Project: Kaggle Capstone Project
- Institution: California State University, East Bay (CSUEB)
- Mentor: Professor Surendra Sarnikar
- Author: Shravan Kumaar Venkatesh Kumar (NetID: GA8997)

## Problem Statement

The project goal is to predict **quarterly company sales** from:

- Company-level financial indicators
- Regional and national economic indicators

The business objective is to generate reliable forecasts that support planning decisions such as inventory, budgeting, operations, and resource allocation.

## Project Objective

Build a robust regression model that minimizes prediction error on unseen data. The evaluation metric is **Mean Absolute Error (MAE)**, chosen because it is directly interpretable in business units.

## Final Model and Metrics

Based on the capstone report, the final selected approach was:

- Model: **LightGBM (boosting_type = gbdt)**
- Tuning: **Optuna (300 trials)**
- Features: Full feature set with engineered interaction and rolling features
- Encoding: Target encoding for categorical variables

### Performance Summary

| Model | Best Validation MAE | Kaggle MAE |
|---|---:|---:|
| LightGBM (gbdt, Optuna tuned) | ~646 | **537.85** |
| XGBoost (Optuna tuned) | ~1115 | 842 |
| CatBoost (Optuna tuned) | ~883 | 927 |

Additional ensemble experiments from the report:

- Weighted ensemble test MAE: 776
- Stacked ensemble Kaggle MAE: 608

## Repository Structure

- `python files/`: training and experimentation notebooks (LightGBM, XGBoost, Random Forest, tuning)
- `Submission/`: generated submission CSVs and exported notebook PDFs
- `train.csv`, `test.csv`, `sample_submission.csv`: core challenge datasets
- `EconomicIndicators.csv`: supporting economic variables
- `Kaggle Capstone Project Report.pdf` and related documents: final report and supporting artifacts
- `guidlines/`: project guideline screenshots and references

## How To Run

```bash
git clone https://github.com/shravankumaar31/b2b-sales-forecasting-ensemble.git
cd b2b-sales-forecasting-ensemble
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy scikit-learn xgboost lightgbm optuna matplotlib seaborn
jupyter notebook
```

Then run notebooks from `python files/` in sequence based on your experiment path.

## Notes

- `.gitignore` excludes `Shravan_GA8997_Kaggle_capstone_Reflection.pptx` because it exceeds GitHub's 100 MB file limit.
- For exact reproducibility, you can add a pinned `requirements.txt` later.
