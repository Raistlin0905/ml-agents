from typing import List, Tuple
from xgboost import XGBRegressor
from Model import Model

from Model import Model


class XGBoostModel(Model[list[float], int]):
    def __init__(self, **kwargs):
        # Allow hyperparameters to be passed optionally
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            **kwargs
        )

    # training method

    def train(self, training_data: List[Tuple[list[float], int]]) -> None:
        """
        training_data = [
            ([feature1, feature2, ...], label),
            ...
        ]
        """
        features, labels = zip(*training_data)
        self.model.fit(list(features), list(labels))

    # prediction method

    def predict(self, x: List[list[float]]) -> List[int]:
        """
        x = list of feature vectors, example:
            [
                [0.12, 0.55, 3.1],
                [0.20, 0.44, 2.9]
            ]
        """
        preds = self.model.predict(x)
        return preds.tolist()
