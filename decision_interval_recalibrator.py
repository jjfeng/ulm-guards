import logging
import numpy as np
import scipy
import scipy.optimize

from dataset import Dataset
from decision_prediction_base_model import DecisionPredictionBaseModel
from common import is_within_interval, get_interval_width

class DecisionIntervalRecalibrator:
    """
    Calculates the conditional coverage of fitted intervals/decision functions
    This is for a single model!
    """
    def __init__(self, fitted_model: DecisionPredictionBaseModel, alpha: float):
        """
        @param alpha: record which (1-alpha) prediction intervals we want to recalib for
        """
        self.fitted_model = fitted_model
        self.alpha = alpha

    def recalibrate(self, recalib_data: Dataset):
        """
        @return a dict of dicts containing mean and se of different coverage probabilities
        """
        accept_probs = self.fitted_model.get_accept_prob(recalib_data.x).flatten()
        alpha_PIs = self.fitted_model.get_prediction_interval(recalib_data.x, self.alpha)
        within_check = is_within_interval(alpha_PIs, recalib_data.y)

        coverage_inference_dict = {}

        cover_and_accept = within_check * accept_probs
        cov_and_accept_mean = np.mean(cover_and_accept)
        print("num obs", recalib_data.num_obs)
        cov_and_accept_var = np.var(cover_and_accept)
        cov_and_accept_se = np.sqrt(cov_and_accept_var/recalib_data.num_obs)

        accept_mean = np.mean(accept_probs)
        accept_probs_var = np.var(accept_probs)
        accept_se = np.sqrt(accept_probs_var/recalib_data.num_obs)

        covariance_off_diag = np.mean(
                accept_probs * cover_and_accept) - cov_and_accept_mean * accept_mean
        covariance_matrix = np.array([
            [cov_and_accept_var, covariance_off_diag],
            [covariance_off_diag, accept_probs_var]])
        grad_divide = np.array([[
            1.0/accept_mean,
            -cov_and_accept_mean/np.power(accept_mean, 2)]])
        delta_method_var = np.matmul(np.matmul(grad_divide, covariance_matrix), grad_divide.T)
        delta_method_se = np.sqrt(delta_method_var/recalib_data.num_obs)[0,0]
        pi_widths = get_interval_width(alpha_PIs)
        logging.info("alpha PI width mean (of accepted) %f", np.sum(pi_widths * accept_probs)/np.sum(accept_probs))
        logging.info("alpha PI width median (of accep5 > 0.5) %f", np.median(pi_widths[accept_probs > 0.5]))

        coverage_inference_dict["accept"] = {
                "mean": accept_mean,
                "se": accept_se}
        coverage_inference_dict["cov_and_accept"] = {
                "mean": cov_and_accept_mean,
                "se": cov_and_accept_se}
        coverage_inference_dict["cov_given_accept"] = {
                "mean": cov_and_accept_mean/accept_mean,
                "se": delta_method_se,
                "covariance": covariance_matrix,
                "num_obs": recalib_data.num_obs}

        print("raw mean", cov_and_accept_mean)
        print("accept mean", accept_mean)
        logging.info("raw mean %f", cov_and_accept_mean)
        logging.info("accept mean %f", accept_mean)
        print("FINAL COVERAGE ESTIMATE", coverage_inference_dict["cov_given_accept"])
        return coverage_inference_dict

    def eval_cov_given_accept(self, data):
        recalib_dict = self.recalibrate(data)
        return recalib_dict["cov_given_accept"]["mean"]
