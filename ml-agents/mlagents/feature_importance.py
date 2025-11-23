import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor


class FeatureImportance:
    def __init__(self, csv_path, model, target="runtime"):
        self.csv_path = csv_path # path to the data set to analyze
        self.model = model # any ML model used to predict SHAP (not ours sepcifically, just any)
        self.target = target # name of the column we want to predict

    # load the data -> split into features +target
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.y = self.df[self.target]
        self.X = self.df.drop(columns=[self.target])

    # train the chosen model on the data (otherwise SHAP won't work)
    def train_model(self):
        self.model.fit(self.X, self.y)

    # SHAP values - tells us how much each feature contributes to the prediction
    def compute_shap(self):
        explainer = shap.TreeExplainer(self.model) #  shap looks inside the XGBoost trees to understand how features influence predictions
        shap_values = explainer.shap_values(self.X) # a matrix: (rows = data samples, columns = features)
        importance = np.abs(shap_values).mean(axis=0) # mean absolute value, which is the standard SHAP importance metric
        self.feature_importances = pd.Series(importance, index=self.X.columns)
        return self.feature_importances.sort_values(ascending=False)

    def select_features(self, top_n=20): # default to 20
        selected_cols = self.feature_importances.nlargest(top_n).index
        return self.df[selected_cols.tolist() + [self.target]]

    def save_cleaned_dataset(self, df, output_path):
        df.to_excel(output_path, index=False)

# here i used XGBRegressor works for well, you guessed it, regression. Also SHAP works best with regression trees.
# it works internally kinda, not a model we use for actually predicting anything


selector = FeatureImportance(
    csv_path="ml-agents/mlagents/sample_runtime_data.csv",
    model=XGBRegressor(),
    target="runtime"
)

selector.load_data()
selector.train_model()

importances = selector.compute_shap()
print(importances)

# for the real dataset, we have to set top_n to however much features we want to keep, it keeps the n most important ones
clean_df = selector.select_features(top_n=4)
selector.save_cleaned_dataset(clean_df, "cleaned_training_data.xlsx")
