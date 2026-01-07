import typing
import scipy.stats as stats

# mann whitney u test is used here: the reasoning goes as follows
# derivative based approach assumes the function shape to look like the log function
# i.e. reward rises rapidly and stagnates presenting asymptotic behavor.
# that assumption is too flakey and therefore a stochastic approach was chosen.
# we dont know the distrubution shape of the reward values so mann whitney is chosen and not the t-test

# null hyp: there is no difference between the samples
# alternative: there is a statistically significant difference


class Reward_Stagnation_Detector:
    """
    checks whether or not a reward data sequence has any major changes windowSize to windowSize multiple times.

    basically, get a window size worth of data from the beginning of the reward data sequence, and
    compare it with the next window size worth of data to see if there is a statiscally significant change,
    the windows shift and the process starts again tallying the number significant change was detected in the sample.
    if there isn't any change for a majority (size of which is controlled by the declarationThresh) of the reward data,
    runCheck() returns true meaning the reward data is stagnating and it is probably a good idea to stop training early.
    instructions:
        - get iterationCount * windowSize worth of reward data samples
        - initialize this object with that data and runCheck
        - if true is returned stop the training
    tuning:
        - if the training stops too early change pvalthresh to 0.05 or 0.01 depending on how long you want the training to go on further for
        - if you want to be surer adjust declarationThresh to be closer to one and/or add more iterations (you therefore need more data)
        - if you want to be even surer you can adjust window size
    note: make sure you get recent reward data: filling up a buffer, running the check, if it fails empty the buffer and fill it again with new data
    """

    def __init__(
        self,
        reward_data: list[float],
        windowSize: int = 1000,
        iterationCount: int = 200,
        pvalthresh: float = 0.05,
        declarationThresh: float = 3 / 5,
    ):
        self.windowSize = windowSize
        self.iters = iterationCount
        self.thresh = pvalthresh
        self.data = list(reward_data)
        self.decThresh = declarationThresh
        assert len(reward_data) == iterationCount * windowSize

    def checkForStagnation(self, prevSamples: list, currentSamples: list) -> bool:
        r = stats.mannwhitneyu(prevSamples, currentSamples)

        return r.pvalue > self.thresh  # decrease thresh to 0.05 or 0.01 to stop later

    def runCheck(self) -> bool:
        cnt = 0
        # iters - 1 compares
        for i in range(self.iters):
            if (i + 2) * self.windowSize <= len(self.data):
                prev = self.data[i * self.windowSize : (i + 1) * self.windowSize]
                curr = self.data[(i + 1) * self.windowSize : (i + 2) * self.windowSize]
                cnt += (
                    1
                    if self.checkForStagnation(prevSamples=prev, currentSamples=curr)
                    else 0
                )
        return (cnt / (self.iters - 1)) >= self.decThresh

    # broken implementation if examinars say no no to scipy

    # def __init__(self, windowSize: int, u_critic: float):
    #     self.windowSize = windowSize
    # def checkForStagnation(self, prevSamples: list, currentSamples: list) -> bool:
    #     assert(prevSamples.count == self.windowSize and currentSamples.count == self.windowSize)
    #     allSamples = np.array(prevSamples + currentSamples)
    #     allSamples = np.sort(allSamples)
    #     ranksUnresolved = enumerate(allSamples, start=1)
    #     ranksResolved = self.resolveTies(ranks=ranksUnresolved)

    #     prevRanks = self.assignRanks(prevSamples, ranksList=ranksResolved)
    #     currentRanks = self.assignRanks(currentSamples, ranksList=ranksResolved)

    #     u_stat_prev = (self.sumRanks(prevRanks)) - ((self.windowSize * (self.windowSize+1)) / 2)
    #     u_stat_curr = (self.sumRanks(currentRanks)) - ((self.windowSize * (self.windowSize+1)) / 2)

    #     u_stat = u_stat_curr if u_stat_curr < u_stat_prev else u_stat_prev

    #     return u_stat > self.u_critic

    # def resolveTies(self, ranks: list[tuple[float, any]]):
    #     for metaIdx in range(len(ranks)):
    #         rank, val = ranks[metaIdx]
    #         for otherMetaIdx in range(len(ranks)):
    #             otherRank, otherVal = ranks[otherMetaIdx]
    #             if otherRank != rank and otherVal == val:
    #                 newRank = (rank + otherRank) / 2
    #                 ranks[metaIdx][0] = newRank
    #                 ranks[otherMetaIdx][0] = newRank

    # def assignRanks(self, unassignedList: list, ranksList: list[tuple[float, any]]) -> list[tuple[float, any]]:
    #     toBeReturned = []
    #     for val in unassignedList:
    #         for rank, value in ranksList:
    #             if val == value:
    #                toBeReturned.append((rank, value))
    #                break

    #     return toBeReturned

    # def sumRanks(self, ranks: list[tuple[float, any]]) -> float:
    #     sum = 0
    #     for rank, _ in ranks:
    #         sum += rank

    #     return sum
