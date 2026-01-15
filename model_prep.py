import pandas as pd
from typing import List, Tuple
from xgboost_model import XGBoostModel

df = pd.read_csv("cleaned_pool.csv")

df = df.drop("total_steps", axis=1) if "total_steps" in df.columns else df

EXCLUDE_COLS = {"total_duration", "total_steps"}

feature_cols = [
    col
    for col in df.columns
    if col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_cols].values.tolist()
y = df["total_duration"].astype(float).tolist()

training_data: List[Tuple[List[float], int]] = list(zip(X, y))

# ---------------

model = XGBoostModel()
model.train(training_data)

# cpu_cores,disk_free_gb,ram_available_bytes,batch_size,cpu_threads,num_epoch,ram_total_bytes,buffer_size,learning_rate_schedule_constant,learning_rate
new_run = [
    [
        0.0,
        100.0,
        33.38525715492659,
        100.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
    ]
]
# XGBoost
pred = model.predict(new_run)[0]
print(f"Predicted total duration: {pred:.2f} seconds")

avg_loss, avg_time, score = model.test(
    validation_data=training_data, folds=10, w_0=0.7, w_1=0.3
)

print(f"Average Loss (MSE): {avg_loss:.4f}")
print(f"RMSE: {avg_loss ** 0.5:.4f}")
print(f"Average Inference Time: {avg_time:.6f} seconds")
print(f"Score: {score:.4f}")
