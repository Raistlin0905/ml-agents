# ML-Agents Training and Analysis Pipeline

**Group 5**
\
*P2-1: Artificial Intelligence & Machine Learning*

This project trains RL agents in headless Unity environments and builds predictive models to understand which hardware and training factors impact performance. Data is collected across multiple machines, analyzed for feature importance, and used to optimize model accuracy through hyperparameter tuning.

For detailed documentation on specific components, see the [blank/](blank/) directory.

## 1. Setup Virtual Environment

Create and activate a Python virtual environment to isolate your Python dependencies for this project.

```bash
python -m venv venv
```

Then activate it:

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Headless Training

Run the Unity environment without graphics and log the results. Replace `3DBall` with your environment name.

```bash
python data_collection/headless_train.py 3DBall --no-graphics
```

**Output:**
- `training_data.csv` — Contains hardware info, training configurations, and results for each run.

## 4. Encode Training Data

Converts numeric features to 0-100, one-hot encodes categorical features, and preserves run_id.

```bash
python data_collection/feature_encoder.py input.csv output.csv class_label
```

**Arguments:**
- `input` — raw training data CSV (`training_data.csv`)
- `output` — encoded CSV ready for ML (`training_data_encoded.csv`)
- `class_label` — column you want to predict (e.g., `total_steps` or `total_duration`)

**Output:**
- `ouput.csv` — Numeric features normalized, categorical features one-hot encoded, ready for ML models.

## 5. Analyze Feature Importance with SHAP

Identifies which features contribute most to predicting your target.

```bash
python feature_importance.py 
```

**Outputs:**
- Console SHAP importance values
- `importance_bar.png` — horizontal bar plot of top features
- `shap_summary.png` — summary dot plot of feature contributions
- `cleaned_pool.csv` — dataset with top N features + target
- `cleaned_pool.xlsx` — Excel version of cleaned dataset

## 6. Run Model Optimization

Trains and evaluates multiple model types (XGBoost, Ridge, Random Forest) across various hyperparameter configurations and top SHAP-selected features.

```bash
python model_optimisation.py
```

**Outputs:**
- Console logs:
  - SHAP scores
  - RMSE, inference time, and score for all models/configs
  - Best model and feature subset
- `model_comparision_results.csv` — full table of all tested configurations

## Acknowledgments
- Built on [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
- Group 5 (AI/ML) team members
- Uses [SHAP](https://github.com/shap/shap), [XGBoost](https://xgboost.readthedocs.io/), and other open-source libraries

## License
This project is licensed under the [LICENSE](LICENSE.md) file.

## Disclaimer
This project utilizes the Unity ML-Agents framework and example environments. Unity Technologies retains all rights to these components, and this project builds upon their work without claiming ownership of their intellectual property. 