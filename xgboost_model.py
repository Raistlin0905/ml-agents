from typing import List, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time

from Model import Model 


class XGBoostModel(Model[list[float], int]):
    """
    XGBoost model
        - train(training_data)
        - predict(x)
        - test_with_data(training_data, folds)
    """

    def __init__(self, **kwargs):
        # Allow hyperparameters to be passed optionally
        self.model = XGBClassifier(
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
        self.model.fit(features, labels)


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


    # k-fold testing

    def test_with_data(self, training_data: List[Tuple[list[float], int]], folds: int = 5):
        """
        Returns:
            avg accuracy, avg inference time, score
        score = 0.7 * accuracy + 0.3 * (1 / inference_time)
        """
        X, y = zip(*training_data)
        X = list(X)
        y = list(y)

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        accuracies = []
        inference_times = []

        for train_idx, test_idx in kf.split(X):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            # train on fold
            self.model.fit(X_train, y_train)

            # measure inference
            start = time.time()
            preds = self.model.predict(X_test)
            end = time.time()

            accuracies.append(accuracy_score(y_test, preds))
            inference_times.append(end - start)

        avg_acc = sum(accuracies) / folds
        avg_time = sum(inference_times) / folds
        score = 0.7 * avg_acc + 0.3 * (1 / avg_time)

        return avg_acc, avg_time, score
