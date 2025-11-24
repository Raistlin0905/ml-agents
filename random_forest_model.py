from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor

from Model import Model


class RandomForestModel(Model[list[float], float]):
    """
    Random Forest Regressor for predicting ML-Agents training times
    """

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


    def train(self, training_data: List[Tuple[list[float], float]]) -> None:
        """
        Format of data: [(features, training_time), ...]
        Example: [([55, 4, 4, 104, 10000], 360.5), ([26, 2, 3, 500, 500000], 1800.2)]
        """
        features, labels = zip(*training_data)
        self.model.fit(features, labels)


    def predict(self, x: List[list[float]]) -> List[float]:
        predictions = self.model.predict(x)
        return predictions.tolist()

