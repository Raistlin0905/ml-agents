import pandas as pd
from typing import List, Tuple
from xgboost_model import XGBoostModel
from feature_importance import FeatureImportance
from xgboost import XGBRegressor
from itertools import product
from Model import RidgeModel
from random_forest_model import RandomForestModel
#return Model based on model_name
def model_factory(config):
    if config["model_name"] == "xgboost_regressor":
        return XGBoostModel(estimators=config["estimators"], max_depth=config["max_depth"])
    elif config["model_name"] == "ridge_regression":
        return RidgeModel(alpha=config["alpha"])
    elif config["model_name"] == "random_forest":
        return RandomForestModel(n_estimator=config["estimators"], max_depth=config["max_depth"], min_sample_split=config["min_sample_split"])
    
class Test_Result:
    def __init__(self, config, result = None, features_considered = None):
        self.config = config
        self.result = result
        self.features_considered = features_considered

#data processing
df = pd.read_csv("cleaned_pool.csv")

df = df.drop("total_steps", axis=1) if "total_steps" in df.columns else df
df = df[df['total_duration'] < 2000]
#feauters available in clean csv
feature_cols = ["cpu_cores", "disk_free_gb", "ram_available_bytes", "batch_size", 
                "cpu_threads", "num_epoch", "ram_total_bytes", "buffer_size", 
                "learning_rate_schedule_constant", "learning_rate"]

#shap scores
selector = FeatureImportance(csv_path="encoded_pool.csv", model=XGBRegressor(), target="total_duration")
selector.load_data()
selector.train_model()
shap_scores = selector.compute_shap()

shap_score_dict = shap_scores.to_dict()

#filter to feautres where k - key is in feautre_col
shap_score_dict = {k : v  for k, v in shap_score_dict.items() if k in feature_cols}
print("-----SHAP_SCORES-----")
print(shap_score_dict) #already sorted
print("-----SHAP_SCORES-----")

#labels
y = df["total_duration"].astype(float).tolist()

#feauture offset, number of 0 = first top feauter considered, 1 first two top features considered, 2 three and so on
feature_offset_options = list(range(len(shap_score_dict)))

#XGBoostRegressor
xgboost_n_estimator_options = [100, 500, 1000] #purely arbitrary, more fine grained adjustments will be made later
xgboost_max_depth = [3, 5, 10]

xgboost_configs = [{"model_type":"xgboost", "model_name" : "xgboost_regressor", "estimators" : estimator, "max_depth" : depth}
                   for estimator, depth in product(xgboost_n_estimator_options, xgboost_max_depth)
                   ]


#Ridge regression
ridge_alpha_options = [0.1, 1, 10]

ridge_configs = [{"model_type":"linear_regression", "model_name" : "ridge_regression", "alpha" : a}  
                 for a in ridge_alpha_options
                 ]

#Random Forest
rf_estimator_count_options = [50, 100, 200]
rf_max_depth_options       = [5, 10, None]
rf_min_sample_split        = [2, 5, 10]
rf_configs = [
    {"model_type":"ensemble", "model_name" : "random_forest", "estimators" : estimator_count, "max_depth" : depth, "min_sample_split" : min_split}
    for estimator_count, depth, min_split in product(rf_estimator_count_options, rf_max_depth_options, rf_min_sample_split)
]
#main loop adding feautures one by one using shap priority
print("Starting comparision, this might take a while....")
all_results = []
for offset in feature_offset_options:
    considered_features = list(shap_score_dict.keys())[:offset+1]
    X = df[considered_features].values.tolist()
    training_data: List[Tuple[List[float], int]] = list(zip(X, y))
    for config in xgboost_configs + ridge_configs + rf_configs:
        model = model_factory(config)
        test_result = model.test(validation_data=training_data, folds=5, w_0=0.999, w_1=0.001)

        result = Test_Result(config=config, result=test_result, features_considered=considered_features)
        all_results.append(result)

print("\n\n-----RESULT-----")
for result in all_results:
    avg_loss, avg_time, score = result.result
    print(f"\nConfig: {result.config}")
    print(f"features_considered: {result.features_considered}")
    print(f"RMSE: {avg_loss ** 0.5:.2f}, Time: {avg_time:.6f}s, Score: {score:.2f}")


#persist results
pd_result_data = []

for result in all_results:
    avg_loss, avg_time, score = result.result
    rmse = avg_loss ** 0.5

    row = {
        "model_type" : result.config["model_type"],
        "model_name" : result.config["model_name"],
        "num_features" : len(result.features_considered),
        "features": "+ ".join(result.features_considered),
        "RMSE": rmse,
        "avg_loss": avg_loss,
        "inference_time": avg_time,
        "score" : score
    }

    for k, v in result.config.items():
        if k not in ["model_type", "model_name"]:
            row[k] = v
        
    pd_result_data.append(row)

results_df = pd.DataFrame(pd_result_data)
results_df.to_csv("model_comparision_results.csv", index=False)
print("saved results :)")

best = min(all_results, key=lambda r: r.result[2]) #min score

avg_loss, avg_time, score = best.result

print("\n\n----BEST MODEL----")
print(f"Model: {best.config['model_name']}")
print(f"Config: {best.config}")
print(f"Features ({len(best.features_considered)}): {best.features_considered}")
print(f"RMSE: {avg_loss ** 0.5:.2f}")
print(f"Inference Time: {avg_time:.6f}s")
print(f"Score: {score:.2f}")