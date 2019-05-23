import numpy as np
from numpy import ndarray
from typing import List, Tuple

from dataset import Dataset
from data_feature_range import FeatureRange
from common import mult_list


class SupportSimSettings:
    """
    Specifies what the support of X looks like
    Can generate data over the support of X uniformly
    """
    def support_unif_rvs(self, size: int):
        raise NotImplementedError("implement me")

    def check_dataset(self, dataset):
        raise NotImplementedError("implement me")

class SupportSimSettingsContinuousMulti(SupportSimSettings):
    """
    Here we specify separate min and max for each dimension
    Specifies what the support of X looks like
    Can generate data over the support of X uniformly
    """
    def __init__(self, num_p: int, min_x: ndarray, max_x: ndarray):
        self.num_p = num_p
        self.min_x = min_x.reshape((1,num_p))
        self.max_x = max_x.reshape((1,num_p))
        #assert min_x.shape == (num_p,)
        #assert max_x.shape == (num_p,)

    def support_unif_rvs(self, n: int):
        """
        @return random vectors drawn from the support uniformly
        """
        return np.random.rand(n, self.num_p) * (self.max_x - self.min_x) + self.min_x

    def check_dataset(self, dataset):
        """
        @return whether all points are within the support
        """
        in_support = np.sum(
                (dataset.x >= self.min_x) * (dataset.x <= self.max_x),
                axis=1) == self.num_p
        print("percent in support", np.sum(in_support) / dataset.num_obs)
        return np.sum(in_support) == dataset.num_obs

    def check_obs_x(self, x):
        """
        @return whether all points are within the support
        """
        num_in_support = np.sum(
                (x >= self.min_x) * (x <= self.max_x),
                axis=1)
        in_support = num_in_support == self.num_p
        return in_support

class SupportSimSettingsContinuous(SupportSimSettings):
    """
    Specifies what the support of X looks like
    Can generate data over the support of X uniformly
    """
    def __init__(self, num_p: int, min_x: float, max_x: float):
        self.num_p = num_p
        self.min_x = min_x
        self.max_x = max_x

    def support_unif_rvs(self, n: int):
        """
        @return random vectors drawn from the support uniformly
        """
        return np.random.rand(n, self.num_p) * (self.max_x - self.min_x) + self.min_x

    def generate_grid(self, mesh_size: float = 0.1):
        """
        @return a mesh grid over the support
        """
        assert self.num_p == 2
        x = np.arange(self.min_x, self.max_x, mesh_size)
        mesh_grid = np.meshgrid(x,x)
        coordinates = np.concatenate([a.reshape((-1,1)) for a in mesh_grid], axis=1)
        return coordinates, mesh_grid

    def check_dataset(self, dataset):
        """
        @return whether all points are within the support
        """
        in_support = np.sum(
                (dataset.x >= self.min_x) * (dataset.x <= self.max_x),
                axis=1) == self.num_p
        return np.sum(in_support) == dataset.num_obs

    def check_obs_x(self, x):
        """
        @return whether all points are within the support
        """
        in_support = np.sum(
                (x >= self.min_x) * (x <= self.max_x),
                axis=1) == self.num_p
        return in_support

class SupportSimSettingsEmpirical(SupportSimSettings):
    """
    Empirical support -- here are just points approximating some distribution
    Specifies what the support of X looks like
    Can generate data over the support of X uniformly
    """
    def __init__(self, points: List[ndarray], scale: float = 1, min_x: float=None, max_x: float=None):
        self.points = np.array(points)
        self.num_points = len(points)
        self.scale = scale
        self.min_x = min_x
        self.max_x = max_x

    def support_unif_rvs(self, n: int):
        """
        @return random vectors drawn from the support uniformly
        """
        idxs = np.random.choice(self.num_points, size=n, replace=True)
        random_points = self.points[idxs]
        perturbs = np.random.normal(scale=self.scale, size=random_points.shape)
        perturbed_points = random_points + perturbs
        perturbed_points = np.minimum(
                np.maximum(perturbed_points, self.min_x),
                self.max_x)
        return perturbed_points

    def check_dataset(self, dataset):
        """
        @return whether all points are within the support
        """
        return True

    def check_obs_x(self, x):
        """
        @return whether all points are within the support
        """
        return True

class SupportSimSettingsComplex(SupportSimSettings):
    """
    Specifies what the support of X looks like
    Can generate data over the support of X uniformly
    """
    def __init__(self, feature_ranges: List[FeatureRange]):
        self.feature_ranges = feature_ranges
        self.num_p = len(feature_ranges)
        self._process_feature_ranges()

    def _process_feature_ranges(self):
        print("process")
        cts_feature_idxs = []
        discrete_feature_idxs = []
        for idx, feat_range in enumerate(self.feature_ranges):
            if feat_range.is_cts:
                cts_feature_idxs.append(idx)
            if feat_range.is_discrete:
                discrete_feature_idxs.append(idx)
        self.cts_feature_idxs = np.array(cts_feature_idxs, dtype=int)
        self.cts_min_x = np.array([self.feature_ranges[idx].min_x for idx in self.cts_feature_idxs])
        self.cts_max_x = np.array([self.feature_ranges[idx].max_x for idx in self.cts_feature_idxs])
        self.discrete_feature_idxs = np.array(discrete_feature_idxs)
        self.discrete_num_vals = np.array([
            self.feature_ranges[idx].num_feature_values for idx in self.discrete_feature_idxs]).reshape((1, -1))

    def support_unif_rvs(self, n: int):
        """
        @return random vectors drawn from the support uniformly
        """
        if not hasattr(self, 'cts_feature_idxs'):
            self._process_feature_ranges()

        rand_features = np.zeros((n, self.num_p))
        # Make the cts ones first
        rand_features[:, self.cts_feature_idxs] = np.random.rand(n, self.cts_feature_idxs.size) * (self.cts_max_x - self.cts_min_x) + self.cts_min_x

        # Now make discrete ones... by drawing a random number in big range, taking the mod and then
        # finding corresponding feature value
        discrete_rand_idxs = np.mod(
                np.random.randint(low=0, high=10000, size=(n, self.discrete_feature_idxs.size)),
                self.discrete_num_vals)
        for old_idx, col_idx in enumerate(self.discrete_feature_idxs):
            rand_features[:, col_idx] = self.feature_ranges[col_idx].feature_values_flat[discrete_rand_idxs[:, old_idx]]
        return rand_features

    def check_obs_x(self, x: ndarray):
        """
        @return whether all points are within the support
        """
        is_good = [
            feat_range.in_set(x[:, idx:idx+1])
            for idx, feat_range in enumerate(self.feature_ranges)
        ]
        is_good = np.all(is_good, axis=0)
        return is_good

    def check_dataset(self, dataset: Dataset):
        """
        @return whether all points are within the support
        """
        return np.all(self.check_obs_x(dataset.x))

    @staticmethod
    def create_from_dataset(xs: ndarray, inflation_factor: float = 0.5):
        """
        @return List[FeatureRange]
        """
        num_p = xs.shape[1]
        return SupportSimSettingsComplex(
                [FeatureRange.create_from_data(xs[:,i], inflation_factor) for i in range(num_p)])

