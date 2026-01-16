import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


class FeatureImportance:
    def __init__(self, csv_path, model, target="runtime"):
        self.csv_path = csv_path  # path to the data set to analyze
        self.model = (
            model  # any ML model used to predict SHAP (not ours sepcifically, just any)
        )
        self.target = target  # name of the column we want to predict

    # load the data -> split into features +target
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.y = self.df[self.target]
        self.X = self.df.drop(columns=[self.target])

    # train the chosen model on the data (otherwise SHAP won't work)
    def train_model(self):
        self.X = self.X.drop(columns=["RunID"], errors="ignore")
        self.X = self.X.drop(columns=["id"], errors="ignore")
        self.X = self.X.drop(columns=["run_id"], errors="ignore")
        self.X = self.X.drop(columns=["total_steps"], errors="ignore")
        self.X = self.X.drop(columns=["time_start"], errors="ignore")
        self.X = self.X.drop(columns=["time_end"], errors="ignore")
        self.model.fit(self.X, self.y)

    # SHAP values - tells us how much each feature contributes to the prediction
    def compute_shap(self):
        self.explainer = shap.TreeExplainer(self.model)  #  shap looks inside the XGBoost trees to understand how features influence predictions
        self.shap_values = self.explainer.shap_values(self.X)  # a matrix: (rows = data samples, columns = features)
        importance = np.abs(self.shap_values).mean(axis=0)  # mean absolute value, which is the standard SHAP importance metric
        self.feature_importances = pd.Series(importance, index=self.X.columns)
        return self.feature_importances.sort_values(ascending=False)

    def select_features(self, top_n=20):  # default to 20
        selected_cols = self.feature_importances.nlargest(top_n).index
        return self.df[selected_cols.tolist() + [self.target]]

    def save_cleaned_dataset(self, df, output_path):
        df.to_excel(output_path, index=False)

    def plot_importance_bar(self, top_n=20, output_path="importance_bar.png"):
        """Bar plot showing importance scores (mean absolute SHAP values)"""
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importances.nlargest(top_n)
        bars = plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        for bar, val in zip(bars, top_features.values):
            width = bar.get_width()
            plt.gca().text(width + max(top_features.values) * 0.01, bar.get_y() + bar.get_height() / 2,f"{val:.4f}", va='center')
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved importance bar plot to {output_path}")
        plt.close()

    def plot_summary(self, top_n=20, output_path="shap_summary.png"):
        """Summary plot showing feature contributions"""
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importances.nlargest(top_n).index
        top_indices = [list(self.X.columns).index(f) for f in top_features]
        shap.summary_plot(self.shap_values[:, top_indices], self.X[top_features], plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to {output_path}")
        plt.close()

# here i used XGBRegressor works for well, you guessed it, regression. Also SHAP works best with regression trees.
# it works internally kinda, not a model we use for actually predicting anything


if __name__ == "__main__":
    selector = FeatureImportance(
        csv_path="encoded_pool.csv", model=XGBRegressor(), target="total_duration"
    )

    selector.load_data()
    selector.train_model()

    importances = selector.compute_shap()
    print(importances)

    selector.plot_importance_bar(top_n=15, output_path="importance_bar.png")
    selector.plot_summary(top_n=15, output_path="shap_summary.png")

    clean_df = selector.select_features(top_n=10)
    selector.save_cleaned_dataset(clean_df, "cleaned_pool.xlsx")

    clean_df.to_csv("cleaned_pool.csv", index=False)

