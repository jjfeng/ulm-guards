from sklearn.base import BaseEstimator
from numpy import ndarray

class DecisionPredictionBaseModel(BaseEstimator):
    def get_accept_prob(self, x: ndarray):
        """
        @param x: covariates
        @return accept prob
        """
        raise NotImplementedError("You need to implement this!")

    def get_prediction_interval(self, x: ndarray, alpha: float =0.1):
        """
        @param x: covariates
        @param alpha: create PI with width (1 - alpha) coverage
        @return prediction interval
        """
        raise NotImplementedError("You need to implement this!")
