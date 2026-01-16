from typing import List, Tuple
from xgboost import XGBRegressor
from Model import Model

from Model import Model


class XGBoostModel(Model[list[float], int]):
    def __init__(self, estimators=1000, max_depth=10, learning_rate=0.01, subsample=0.8, colsample_bytree=0.9, **kwargs):
        # Allow hyperparameters to be passed optionally
        self.model = XGBRegressor(
            n_estimators=estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
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
