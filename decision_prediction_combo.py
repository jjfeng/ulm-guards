import numpy as np
from sklearn.ensemble import IsolationForest

class AcceptAllPredictionModel:
    def __init__(self, interval_nn):
        self.nn = interval_nn
        self.get_univar_mapping = self.get_accept_prob
        self.score = self.nn.score

    def get_accept_prob(self, x):
        return np.ones((x.shape[0], 1))

class DecisionPredictionModel:
    def __init__(self, interval_nn, cost_decline):
        self.nn = interval_nn
        self.cost_decline = cost_decline
        self.get_univar_mapping = self.nn.get_univar_mapping

    def get_accept_prob(self, x):
        return np.array(self.nn.get_univar_mapping(x) < self.cost_decline, dtype=int)

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = self.nn.get_prediction_loss_obs(x, y)
        print("pred loss dist", np.median(pred_loss), np.mean(pred_loss), np.min(pred_loss), np.max(pred_loss))
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))

class EntropyOutlierPredictionModel:
    def __init__(self, interval_nn, cost_decline, eps: float = 0):
        self.nn = interval_nn
        self.cost_decline = cost_decline
        self.get_univar_mapping = lambda x: -self.get_accept_prob(x)
        self.od_model = OutlierPredictionModel(interval_nn, cost_decline, eps)

    def fit_decision_model(self, X):
        """
        Fits an IsolationForest outlier classifier based on training data X
        eps parameter in (0,1] controls proportion of observations removed
        """
        self.od_model.fit_decision_model(X)

    def get_accept_prob(self, x):
        short_enuf = np.array(self.nn.get_univar_mapping(x) < self.cost_decline, dtype=int)
        return short_enuf * self.od_model.get_accept_prob(x)

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = self.nn.get_prediction_loss_obs(x, y)
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))

class OutlierPredictionModel:
    """
    Dummy class
    """
    def __init__(self, interval_nn, cost_decline, eps: float = 0):
        self.nn = interval_nn
        self.cost_decline = cost_decline
        self.eps = eps
        self.iso_forest = IsolationForest(behaviour='new', contamination='auto')
        self.fitted_scores = []
        self.get_univar_mapping = lambda x: -self.score_samples(x)

    def score_samples(self, X):
        return 1 + self.iso_forest.score_samples(X).reshape((X.shape[0], 1))

    def fit_decision_model(self, X):
        """
        Fits an IsolationForest outlier classifier based on training data X
        eps parameter in (0,1] controls proportion of observations removed
        """
        self.iso_forest.fit(X)
        # outlier_scores closer to 0 are more likely to be outliers
        self.fitted_scores = self.score_samples(X)

    @property
    def thresh(self):
        # Classify the bottom eps percent of scores as outliers
        return np.quantile(self.fitted_scores, self.eps)

    def get_accept_prob(self, X):
        """
        Uses iso_forest and thresh to determine which observations to remove.
        """
        outlier_scores = self.score_samples(X)
        accept = outlier_scores > self.thresh
        return accept

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = self.nn.get_prediction_loss_obs(x, y)
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))
