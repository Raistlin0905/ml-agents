from abc import ABC as AbstractBaseClass, abstractmethod
from typing import Any, Callable, TypeVar, Generic
from numpy import array_split
import time
import random
import warnings
import numpy

Features = TypeVar("Features")
Label = TypeVar("Label")

warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)


# this is the basic class of a ml model.
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

    def test(
        self,
        validation_data: list[tuple[Features, Label]],
        folds: int,
        w_0: float,
        w_1: float,
    ) -> tuple[float, float, float]:
        """
        Runs benchmarking system on the model.

        folds -- number of folds in K-fold cross validation.

        returns:
            a tuple of three values (in order):
                - a = Avg. loss across all **folds**
                - t = Avg. inference time per sample
                - s = score =  w_0 * a + w_1 * (1/t); where w_0, w_1 are weight coefficients for a and t respectively.
        """
        cv = KCrossValidation(
            validation_data=validation_data,
            folds=folds,
            model=self,
            loss_function=KCrossValidation.SRLossStartegy,
        )
        a, t = cv.run()

        s = w_0 * a + w_1 * (1 / t)

        return (a, t, s)


class KCrossValidation(Generic[Features, Label]):
    def __init__(
        self,
        validation_data: list[tuple[Features, Label]],
        folds: int,
        model: Model[Features, Label],
        loss_function: Callable[[Label, Label], float],
    ):
        # self.data = random.shuffle(validation_data)
        self.data = validation_data.copy()
        random.shuffle(self.data)
        self.folds = folds
        self.model = model
        self.loss = loss_function

    def run(self) -> tuple[float, float]:
        partitions = array_split(self.data, self.folds)  # get data in k-fold partions
        totalLoss = 0
        totalInferenceDurations = 0

        for i in range(len(partitions)):
            partitionsCpy = partitions.copy()
            testPartition = partitionsCpy.pop(i)
            # concatinate partitions to make them one array again
            trainingPartitons = [item for part in partitionsCpy for item in part]

            self.model.train(trainingPartitons)
            testFeatures = list()
            testActualLabels = list()

            for j in range(len(testPartition)):
                features, label = testPartition[j]
                testFeatures.append(features)
                testActualLabels.append(label)

            # since list instantiation has overhead, to keep the estimates of the real inference time avg.
            # we predict over the whole partition and not a singular data point and collect the inference time for
            # the whole partition. ammoritizing
            # python list creation overheads
            startTime = time.time()
            testPredictionLabels = self.model.predict(
                testFeatures
            )  # list of predictions by the model
            endTime = time.time()

            locPartitionInferenceTime = (
                endTime - startTime
            )  #!!FOR THE WHOLE PARTITION (not individual data points)

            sizeOfTestData = len(testPredictionLabels)
            # assuming the ordering of the predicted labels matches the order of the
            # actual labels
            assert len(testPredictionLabels) == len(testActualLabels)
            locTotalLoss = 0
            for k in range(sizeOfTestData):
                locTotalLoss += self.loss(testActualLabels[k], testPredictionLabels[k])
            totalLoss += locTotalLoss
            totalInferenceDurations += locPartitionInferenceTime

        avgInferenceTime = totalInferenceDurations / len(self.data)
        avgLoss = totalLoss / len(self.data)
        return (avgLoss, avgInferenceTime)

    # SR = squared residual, assumes label is subtractable, which should be since it is in units of seconds
    @staticmethod
    def SRLossStartegy(actual: Label, predicted: Label) -> float:
        return (actual - predicted) ** 2
