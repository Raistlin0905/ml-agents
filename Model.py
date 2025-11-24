
from abc import ABC as AbstractBaseClass, abstractmethod
from typing import Any, TypeVar, Generic

Features = TypeVar("Features")
Label = TypeVar("Label")

#this is the basic class of a ml model.
class Model(AbstractBaseClass, Generic[Features, Label]):
    @abstractmethod
    def predict(self, x: list[Features]) -> list[Label]:
        """
        predict result from features

        returns:
            A **list** of predictions. Each index corresponds to some index in the features list
        """
        pass
    @abstractmethod
    def train(self, training_data: list[tuple[Features, Label]]) -> None:
        """
        Trains the model based on a training data list of tuples.

        training_data -- list of tuples where each element has two elements,
        the features and labels **in order**.

        Example (training data for weight predictor):

            training_data =
            [
                ([174cm, middle_eastern], 75kg),
                ([180cm, white], 83kg),
                ([155cm, south_east_asian], 55kg)
            ]
        """
        pass
    def test(self, folds: int) -> tuple[float, float, float]:
        """
        Runs benchmarking system on the model.

        folds -- number of folds in K-fold cross validation.
        
        returns:
            a tuple of three values (in order):
                - a = Avg. accuracy across all **folds**
                - t = Avg. inference time
                - s = score =  w_0 * a + w_1 * (1/t); where w_0, w_1 are weight coefficients for a and t respectively.
        """
        #TODO: complete this function to implement cross validation to return accuracy and return avg inference time
        raise NotImplemented("test function in Model class is not implemented yet")