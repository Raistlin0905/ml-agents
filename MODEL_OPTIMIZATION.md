# Model Optimization

This module trains and evaluates multiple machine learning models across different hyperparameter configurations to identify the best predictor of training duration. It uses SHAP-based feature selection to iteratively add the most important features, then compares XGBoost, Ridge Regression, and Random Forest models.

## Prerequisites

- `cleaned_pool.csv` — Dataset with top SHAP-selected features
- `encoded_pool.csv` — Encoded training data used to compute SHAP scores
- Dependencies installed (XGBoost, scikit-learn, pandas)

## How to Run

```bash
python model_optimization.py
```

The script will:
1. Load the cleaned dataset and filter outliers (training duration < 2000 seconds)
2. Compute SHAP scores from the encoded data to determine feature importance
3. Train models with 1 feature, then 2, then 3, etc., following SHAP priority order
4. Test each configuration using 10-fold cross-validation
5. Output results to console and save detailed results to CSV

## Configuration

The script tests the following hyperparameter combinations:

**XGBoost Regressor:**
- Estimators: 100, 500, 1000
- Max depth: 3, 5, 10

**Ridge Regression:**
- Alpha: 0.1, 1, 10

**Random Forest:**
- Estimators: 50, 100, 200
- Max depth: 5, 10, None
- Min sample split: 2, 5, 10

**Cross-Validation Scoring:**
- Folds: 5
- Loss metric: Squared Residual (SR)
- Score formula: `w_0 * avg_loss + w_1 * (1/avg_time)` where w_0=0.999, w_1=0.001
  - Prioritizes prediction accuracy (loss) over inference speed

## Outputs

**Console Output:**
- SHAP scores for all features (ranked by importance)
- For each model/config combination:
  - Config details (model type, hyperparameters)
  - Features considered
  - RMSE (Root Mean Squared Error)
  - Average inference time per sample
  - Overall score
- Best model summary with all metrics

**Files Created:**
- `model_comparision_results.csv` — Complete table of all tested configurations with:
  - Model type and name
  - Number of features used
  - Feature list
  - RMSE, average loss, inference time, and score
  - All hyperparameter values

## Understanding the Results

**RMSE (Root Mean Squared Error):** Lower is better. Measures prediction accuracy in seconds.

**Inference Time:** Average time per prediction. Lower is better for production use.

**Score:** Combined metric balancing accuracy and speed. Lower is better.

**Best Model:** The script identifies and prints the model with the lowest score, along with its feature set and exact hyperparameters. Use this as your production model.

## Example Output

```
-----SHAP_SCORES-----
{'cpu_cores': 143.34, 'disk_free_gb': 136.36, 'ram_available_bytes': 113.10, ...}
-----SHAP_SCORES-----

Config: {'model_type': 'xgboost', 'model_name': 'xgboost_regressor', 'estimators': 500, 'max_depth': 5}
features_considered: ['cpu_cores', 'disk_free_gb', 'ram_available_bytes']
RMSE: 156.23, Time: 0.000234s, Score: 156.24

...

----BEST MODEL----
Model: random_forest
Config: {'model_type': 'ensemble', 'model_name': 'random_forest', 'estimators': 100, 'max_depth': 10, 'min_sample_split': 5}
Features (3): ['cpu_cores', 'disk_free_gb', 'ram_available_bytes']
RMSE: 142.15
Inference Time: 0.000189s
Score: 142.16
```

## Notes

- The script filters out training sessions with duration > 2000 seconds (assumed outliers)
- Features are added in order of SHAP importance, so earlier feature combinations are tested first
- Inference time is measured per-sample but amortized across a partition to account for Python overhead
- Results are sorted by score; manually inspect the CSV for alternative models
