import numpy as np
import scipy
import scipy.optimize

from dataset import Dataset
from common import is_within_interval

class DecisionIntervalAggregator:
    """
    DEPRECATED!!!

    Aggregator of fitted models and their prediction intervals
    Calculates the conditional coverage of these aggregate intervals using the coverage probs from each
        fitted model.
    """
    def __init__(self, fitted_models, alpha, inference_dicts):
        self.fitted_models = fitted_models
        self.alpha = alpha
        self.inference_dicts = inference_dicts
        if inference_dicts is not None:
            assert len(fitted_models) == len(inference_dicts)

    def add_model(self, model, inf_dict):
        self.fitted_models += [model]
        self.inference_dicts += [inf_dict]

    @property
    def num_models(self):
        return len(self.fitted_models)

    def calc_agg_cover_given_accept(self, ci_alpha):
        """
        Return estimate and CI with coverage at least 1 - ci_alpha
        """
        cov_and_accept_agg = np.mean([inf_dict["cov_and_accept"]["mean"] for inf_dict in self.inference_dicts])
        accept_agg = np.mean([inf_dict["accept"]["mean"] for inf_dict in self.inference_dicts])
        agg_cov_given_accept_mean = cov_and_accept_agg/accept_agg

        agg_variance_bound = np.mean([np.power(inf_dict["cov_and_accept"]["se"], 2) for inf_dict in self.inference_dicts])

        variances = np.array([np.power(inf_dict["cov_and_accept"]["se"], 2) for inf_dict in self.inference_dicts])
        def diff_subg_bound_alpha(t):
            def subg_upper_bound(tau):
                return np.mean(np.exp(variances * np.power(tau, 2)/2))/np.exp(t * tau * accept_agg)
            min_subg_upper_bound = scipy.optimize.minimize(subg_upper_bound, x0 = 0)
            return 2 * min_subg_upper_bound.fun - ci_alpha

        root_solution = scipy.optimize.root_scalar(diff_subg_bound_alpha, bracket=[0,1])
        ci_radius = root_solution.root
        assert root_solution.converged

        print("inaccurate ci radius", np.sqrt(agg_variance_bound) * 1.96)
        print("my subg ci radius", ci_radius)
        ci_upper = min(1.0, agg_cov_given_accept_mean + ci_radius)
        ci_lower = max(0.0, agg_cov_given_accept_mean - ci_radius)
        return {
                "mean": agg_cov_given_accept_mean,
                "ci": (ci_lower, ci_upper)
                }

    def aggregate_prediction_intervals(self, xs):
        probabilistic_intervals = []
        for model_idx, fitted_model in enumerate(self.fitted_models):
            model_accept_probs = fitted_model.get_accept_prob(xs)
            alpha_PIs = fitted_model.get_prediction_interval(xs, self.alpha)
            prob_and_interval = np.concatenate([model_accept_probs, alpha_PIs], axis=1)
            probabilistic_intervals.append([prob_and_interval])
        all_prob_and_intervals = np.concatenate(probabilistic_intervals, axis=0)
        return all_prob_and_intervals

    def eval_cov_given_accept(self, data: Dataset):
        """
        @param data: Dataset
        @return conditional coverage of aggregate intervals
        """
        accept_probs_all = [
            fitted_model.get_accept_prob(data.x) for fitted_model in self.fitted_models]
        all_accept_probs = np.concatenate([v.reshape((-1, 1)) for v in accept_probs_all], axis=1)
        accept_probs_mean = np.mean(all_accept_probs,  axis=1)
        accept_probs_sum = np.sum(all_accept_probs, axis=1)

        indiv_accepts = []
        indiv_cov_given_accept = []
        total_within_checks = 0
        for model_idx, fitted_model in enumerate(self.fitted_models):
            model_accept_probs = accept_probs_all[model_idx] #[accept_mask]
            alpha_PIs = fitted_model.get_prediction_interval(data.x, self.alpha)
            within_check = is_within_interval(alpha_PIs, data.y)
            cov_and_accepted_indiv = within_check.flatten() * model_accept_probs.flatten()
            total_within_checks += cov_and_accepted_indiv
            indiv_cov_given_accept.append(np.sum(cov_and_accepted_indiv)/np.sum(model_accept_probs))
            indiv_accepts.append(np.mean(model_accept_probs))
        mean_within_checks = total_within_checks/self.num_models
        se_covered_and_accept = np.sqrt(np.var(mean_within_checks)/data.num_obs)
        mean_covered_and_accept = np.mean(mean_within_checks)
        mean_covered_given_accept = mean_covered_and_accept/np.mean(accept_probs_mean)
        se_covered_given_accept = se_covered_and_accept/np.mean(accept_probs_mean)

        #accept_mask = np.random.binomial([1] * data.num_obs, accept_probs_mean).astype(bool)
        #total_within_checks = 0
        #for model_idx, fitted_model in enumerate(self.fitted_models):
        #    model_accept_probs = accept_probs_all[model_idx][accept_mask]
        #    alpha_PIs = fitted_model.get_prediction_interval(data.x[accept_mask,:], self.alpha)
        #    within_check = is_within_interval(alpha_PIs, data.y[accept_mask,:])
        #    total_within_checks += within_check.flatten() * model_accept_probs.flatten()/accept_probs_sum[accept_mask].flatten()
        #mean_covered_given_accept = np.mean(total_within_checks)
        #print(mean_covered_given_accept1, mean_covered_given_accept)

        inf_dict = {
                "accept": {
                    "mean": np.mean(accept_probs_mean),
                    "individual": indiv_accepts},
                "cov_given_accept": {
                    "mean": mean_covered_given_accept,
                    "se": se_covered_given_accept,
                    "individual": indiv_cov_given_accept}
                }
        return inf_dict
