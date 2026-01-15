# Feature Importance with SHAP
SHAP (SHapley Additive exPlanations) is a model-agnostic method for explaining predictions of machine-learning models. It assigns each input feature a contribution value that reflects how much that feature influences the model’s output.

## Why we use SHAP?
At the data collection stage, we do not know in advance which hardware, configuration, or training metrics are most informative for predicting training runtime. Manually selecting features would risk introducing human bias and prematurely discarding useful information. SHAP can capture non-linear effects and feature interactions.

We therefore use SHAP to:
- Quantify the importance of each feature in predicting runtime
- Identify features that consistently contribute little or nothing to the prediction
- Reduce the feature space in a data-driven way before training our final models

In this project, SHAP is used only for feature weighting and selection, not as part of the final runtime prediction model.

## Data preparation
Before running SHAP, all features must be numeric. Thus, the dataset must be encoded using .... (see above) before giving it to SHAP.

## Install the required packages
Open terminal / PowerShell in the folder where the script is.
Run this:
```
python -m pip install pandas numpy shap xgboost
```

## How to use SHAP script
It trains a simple XGBoost regression model, computes SHAP feature importances, prints a ranking, and saves a reduced dataset with only the top-N most important features.

### 1. Change the CSV path to the target one

```
csv_path="ml-agents/mlagents/sample_runtime_data.csv",
```
Replace with your file, for example:
```
csv_path="data/3dball_joined_inner.csv",
```

### 2. Change the target column name (if needed)

```
target="runtime"
```
If your target column is called something else (e.g. duration_seconds), change it:
```
target="duration_seconds"
```

### 3. Change how many features you want to keep
```
clean_df = selector.select_features(top_n=4)
```
If you want the top 20 features:
```
clean_df = selector.select_features(top_n=20)
```

### 4. Change output filename (optional)
```
selector.save_cleaned_dataset(clean_df, "cleaned_training_data.csv")
```
Example:
```
selector.save_cleaned_dataset(clean_df, "cleaned_3dball.csv")
```

### 5. Run the script
In terminal:
```
python feature_importance.py
```