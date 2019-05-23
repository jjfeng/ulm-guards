import numpy as np
from numpy import ndarray
import scipy.stats

from common import is_integer

class FeatureRange:
    """
    Specifies possible values for a feature
    """
    def in_set(self, x: ndarray):
        raise NotImplementedError("please implement")

    def generate_unif_rv(self, size_n: int):
        raise NotImplementedError("please implement")

    @staticmethod
    def create_from_data(feature_vec: ndarray, inflation_factor: float = 0.5):
        if np.all(is_integer(feature_vec)):
            return IntegerFeatureRange(np.unique(feature_vec.astype(int)))
        else:
            min_val = np.min(feature_vec)
            max_val = np.max(feature_vec)
            width = max_val - min_val
            inflated_min = min_val - width * inflation_factor
            inflated_max = max_val + width * inflation_factor
            return ContinuousFeatureRange(inflated_min, inflated_max)

class ContinuousFeatureRange(FeatureRange):
    def __init__(self, min_x: float, max_x: float):
        self.min_x = min_x
        self.max_x = max_x

    @property
    def is_cts(self):
        return True

    @property
    def is_discrete(self):
        return False

    def in_set(self, x: ndarray):
        assert x.shape[1] == 1
        in_support = (x >= self.min_x) * (x <= self.max_x)
        return in_support.flatten()

    def generate_unif_rv(self, size_n: int):
        return np.random.rand(size_n) * (self.max_x - self.min_x) + self.min_x

class IntegerFeatureRange(FeatureRange):
    def __init__(self, feature_values: ndarray):
        """
        @param feature_values: array with possible feature values
        """
        self.feature_values = feature_values.reshape((1,-1))
        self.feature_values_flat = feature_values.flatten()
        self.num_feature_values = feature_values.size

    @property
    def is_cts(self):
        return False

    @property
    def is_discrete(self):
        return True

    def in_set(self, x: ndarray):
        assert x.shape[1] == 1
        in_set_res = np.sum(x == self.feature_values, axis=1) >= 1
        return in_set_res.flatten()

    def generate_unif_rv(self, size_n: int):
        selected_idx = np.random.randint(low=0, high=self.num_feature_values, size=size_n)
        rand_vals = self.feature_values_flat[selected_idx]
        return rand_vals
