from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from Model import Model

TARGET_COLUMN = "total_duration"


class RandomForestModel(Model[list[float], float]):
    """Random Forest Regressor for predicting ML-Agents training times. Expects cleaned data"""

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            **kwargs
        )
        self.feature_columns = None

    def train(self, training_data: List[Tuple[list[float], float]]) -> None:
        """
        Format of data: [(features, training_time), ...]
        """
        features, labels = zip(*training_data)
        self.model.fit(features, labels)

    def predict(self, x: List[list[float]]) -> List[float]:
        predictions = self.model.predict(x)
        return predictions.tolist()

    def train_from_csv(self, csv_path: str) -> None:
        """Load CSV and train in one step."""
        data = self.load_from_csv(csv_path)
        self.train(data)

    def load_from_csv(self, csv_path: str) -> List[Tuple[list[float], float]]:
        df = pd.read_csv(csv_path)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV")

        target = df[TARGET_COLUMN]
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        self.feature_columns = feature_cols
        features_df = df[feature_cols]

        training_data = []
        for idx in range(len(features_df)):
            features = features_df.iloc[idx].tolist()
            label = float(target.iloc[idx])
            training_data.append((features, label))

        return training_data
