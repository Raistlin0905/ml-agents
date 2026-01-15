import pandas as pd
from typing import List, Tuple
from xgboost_model import XGBoostModel

df = pd.read_csv("cleaned_pool.csv")

df = df.drop("total_steps", axis=1) if "total_steps" in df.columns else df

feature_cols = ["disk_free_gb", "ram_available_bytes", "batch_size"]
X = df[feature_cols].values.tolist()

threshold = 400
y = df["total_duration"].astype(float).tolist()


training_data: List[Tuple[List[float], int]] = list(zip(X, y))

# ---------------

model = XGBoostModel()
model.train(training_data)

# cpu_cores,disk_free_gb,ram_available_bytes,batch_size,cpu_threads,num_epoch,ram_total_bytes,buffer_size,learning_rate_schedule_constant,learning_rate
new_run = [[4, 2, 22, 32, 4, 5, 3, 33, 0.0, 22, 310]]
# XGBoost
pred = model.predict(new_run)[0]
print(f"Predicted total duration: {pred:.2f} seconds")
