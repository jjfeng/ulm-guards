import numpy as np
import scipy
import scipy.linalg
import scipy.stats

from decision_interval_aggregator import DecisionIntervalAggregator
from common import get_normal_ci

class DecisionIntervalIndptAggregator(DecisionIntervalAggregator):
    """
    Aggregator of fitted models and their prediction intervals
    Assumes the estimates are independently derived
    """
    def calc_agg_cover_given_accept(self, ci_alpha: float):
        """
        Return estimate and CI with coverage at least 1 - ci_alpha
        """
        num_estimates = len(self.inference_dicts)
        #cov_and_accept_agg = np.mean([inf_dict["cov_and_accept"]["mean"] for inf_dict in self.inference_dicts])
        cov_and_accept_agg = np.sum([inf_dict["cov_and_accept"]["mean"] for inf_dict in self.inference_dicts])
        #accept_agg = np.mean([inf_dict["accept"]["mean"] for inf_dict in self.inference_dicts])
        accept_agg = np.sum([inf_dict["accept"]["mean"] for inf_dict in self.inference_dicts])
        avg_accept = np.mean(accept_agg)
        agg_cov_given_accept_mean = cov_and_accept_agg/accept_agg

        #variances = np.array([np.power(inf_dict["cov_and_accept"]["se"], 2) for inf_dict in self.inference_dicts])
        #agg_se = np.sqrt(np.sum(variances))/num_estimates/accept_agg

        block_diag_covariance = scipy.linalg.block_diag(
                *[inf_dict["cov_given_accept"]["covariance"] for inf_dict in self.inference_dicts])
        tot_numerator = cov_and_accept_agg
        tot_denom = accept_agg
        delta_gradient = np.array([1/tot_denom, -tot_numerator/np.power(tot_denom, 2)] * self.num_models).reshape((-1, 1))
        covariance_cov_given_accept = np.matmul(np.matmul(delta_gradient.T, block_diag_covariance), delta_gradient)
        recalib_num_obs = self.inference_dicts[0]["cov_given_accept"]["num_obs"]
        agg_se = np.sqrt(covariance_cov_given_accept/recalib_num_obs)[0,0]


        #variances = np.array([np.power(inf_dict["cov_given_accept"]["se"], 2) for inf_dict in self.inference_dicts])
        #variance_weights = np.array(
        #        [inf_dict["accept"]["mean"]/avg_accept for inf_dict in self.inference_dicts])
        #agg_variance = np.sum(variance_weights * variances)/self.num_models
        #agg_se = np.sqrt(agg_variance)
        #print("indpttn agg se", agg_se)

        ci = get_normal_ci(
                {"mean": agg_cov_given_accept_mean, "se": agg_se},
                ci_alpha=ci_alpha,
                min_lower=0, max_upper=1)
        return {
                "mean": agg_cov_given_accept_mean,
                "ci": ci
                }
