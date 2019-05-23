import numpy as np
from numpy import ndarray
import scipy.stats

import data_gen_funcs
import data_gen_funcs_bernoulli
import data_gen_funcs_multinomial
from data_feature_range import FeatureRange
from common import get_normal_dist_entropy
from dataset import Dataset
from support_sim_settings import SupportSimSettingsContinuous

class DataGenerator:
    """
    Simulation engine
    """
    def __init__(self,
            sim_func_form: str,
            sim_func_name: str,
            num_p: int,
            num_classes: int =1,
            noise_sd: float =0,
            std_dev_x: float =1,
            max_x: float =1):
        self.num_p = num_p
        self.std_dev_x = std_dev_x
        self.num_classes = num_classes
        self.max_x = max_x
        self.min_x = -max_x
        self.noise_sd = noise_sd
        self.sim_func_form = sim_func_form
        if sim_func_form == "gaussian":
            self.mu_func = getattr(data_gen_funcs, sim_func_name + "_mu")
            self.raw_sigma_func = getattr(data_gen_funcs, sim_func_name + "_sigma")
        elif sim_func_form == "bernoulli":
            self.mu_func = getattr(data_gen_funcs_bernoulli, sim_func_name + "_mu")
        elif sim_func_form == "multinomial":
            self.mu_func = getattr(data_gen_funcs_multinomial, sim_func_name + "_mu")
        else:
            print(sim_func_form)
            raise ValueError("huh?")

    def sigma_func(self, xs: ndarray):
        """
        @return sigma when Y|X is gaussian
        """
        if self.sim_func_form == "gaussian":
            return self.noise_sd * self.raw_sigma_func(xs)
        elif self.sim_func_form == "bernoulli":
            raise ValueError("sure?")

    def entropy_func(self, xs: ndarray):
        """
        @return sigma when Y|X is gaussian
        """
        if self.sim_func_form == "gaussian":
            sigma = self.noise_sd * self.raw_sigma_func(xs)
            return get_normal_dist_entropy(sigma)
        elif self.sim_func_form == "bernoulli":
            p = self.mu_func(xs)
            return -p * np.log(p) - (1 - p) * np.log(1 - p)
        else:
            p = self.mu_func(xs)
            return np.sum(-p * np.log(p), axis=1)

    def create_data(self, num_obs: int, seed:int = None):
        """
        @param num_obs: number of observations
        @param seed: if given, set the seed before generating data

        @param tuple with Dataset, SupportSimSettingsContinuous
        """
        if seed is not None:
            np.random.seed(seed)
        dataset = self._create_data(num_obs)
        support_sim_settings = self._create_support_sim_settings()
        return dataset, support_sim_settings

    def create_data_given_x(self, xs: ndarray):
        """
        For the given Xs, generate responses and dataset
        regression-type data only
        @return Dataset
        """
        size_n = xs.shape[0]
        mu_true = self.mu_func(xs)
        if len(mu_true.shape) == 1:
            mu_true = np.reshape(mu_true, (size_n, 1))
        if self.sim_func_form == "gaussian":
            sigma_true = np.reshape(self.sigma_func(xs), (size_n, 1))

            true_distribution = scipy.stats.norm(mu_true, sigma_true)
            y = true_distribution.rvs(size=mu_true.shape)
            true_prob = true_distribution.pdf(y)
        elif self.sim_func_form == "bernoulli":
            true_distribution = scipy.stats.binom(n=1, p=mu_true)
            y = true_distribution.rvs(size=mu_true.shape)
            true_prob = true_distribution.pmf(y)
        elif self.sim_func_form == "multinomial":
            # We have to do per row because multinomial is not nice and doesn't take in
            # 2D probability matrices
            all_ys = []
            all_probs = []
            for i in range(mu_true.shape[0]):
                mu_row = mu_true[i,:]
                true_distribution = scipy.stats.multinomial(n=1, p=mu_row)
                y = true_distribution.rvs(size=1)
                all_ys.append(y)
                true_prob = true_distribution.pmf(y)
                all_probs.append(true_prob)
            y = np.vstack(all_ys)
            true_prob = np.vstack(all_probs)

        # Print entropy of Y|X for fun
        entropies = self.entropy_func(xs)
        print("ENTROPY", np.mean(entropies), np.var(entropies))

        return Dataset(xs, y, true_pdf=true_prob, num_classes=self.num_classes)

    def generate_x(self, size_n: int, buffer_factor: int = 10):
        """
        Generates x from a gaussian distribution
        """
        xs = np.random.randn(size_n * buffer_factor, self.num_p) * self.std_dev_x
        # Only keep the ones within the support
        in_support = np.sum(
                (xs >= self.min_x) * (xs <= self.max_x),
                axis=1) == self.num_p
        xs_in_support = xs[in_support,:]
        assert np.sum(in_support) >= size_n
        return xs_in_support[:size_n, :]

    def get_x_pdf(self, xs: ndarray):
        """
        @return pdf of X
        """
        all_pdfs = 1
        for i in range(xs.shape[1]):
            all_pdfs *= scipy.stats.norm.pdf(xs[:,i], loc=0, scale=self.std_dev_x)
        return all_pdfs

    def get_prediction_interval(self, xs: ndarray, alpha: float):
        mus = self.mu_func(xs).reshape((-1,1))
        sigmas = self.sigma_func(xs).reshape((-1,1))
        z_factor = scipy.stats.norm.ppf(1 - alpha/2)
        lower = mus - z_factor * sigmas
        upper = mus + z_factor * sigmas
        return np.concatenate([lower, upper], axis=1)

    def _create_data(self, size_n: int):
        """
        regression-type data only
        @return Dataset
        """
        # Generate some x's from a gaussian distribution
        data_gen_xs = self.generate_x(size_n)
        return self.create_data_given_x(data_gen_xs)

    def _create_support_sim_settings(self):
        return SupportSimSettingsContinuous(self.num_p, self.min_x, self.max_x)
