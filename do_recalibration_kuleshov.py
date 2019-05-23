"""
Do recalibration of the given intervals using method of Kuleshov et al (2018)
"""
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np

from common import pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci, load_model
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        default="_output/data_split.pkl")
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/recalibrated_coverages.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/recalibrated_log.txt")
    parser.add_argument('--alphas',
        type=str,
        help="""
        Specifies which alpha values to calibrate for
        comma separated list
        """,
        default="0.05,0.1,0.2")
    parser.set_defaults()
    args = parser.parse_args()
    args.alphas = process_params(args.alphas, float)
    return args

def fit_isotonic(fitted_model, recalib_data):
    """
    @return IsotonicRegression object for recalibration of intervals
    """
    # For every x, use fitted model to get predicted y|x
    pred_dists = fitted_model.get_prediction_dist(recalib_data.x)
    accept_probs = fitted_model.get_accept_prob(recalib_data.x).flatten()

    if np.mean(np.isfinite(accept_probs)) < 0.5:
        return None

    # Predicted cdf-s on recalibration data
    cdf_fit = [0] * len(pred_dists)
    recalib_y = recalib_data.y
    for i in range(len(pred_dists)):
        yi = recalib_y[i][0]
        cdf_fit[i] = norm.cdf(yi, pred_dists[i][0], pred_dists[i][1])
    cdf_fit = np.array(cdf_fit)
    # Empirical cdf-s of recalibration data
    p_hat = [np.mean(cdf_fit <= obs_cdf_fit) for obs_cdf_fit in cdf_fit]
    p_hat = np.array(p_hat)

    # Fit isotonic regression
    ir = IsotonicRegression()
    ir.fit(cdf_fit, p_hat, sample_weight=accept_probs)

    return ir

def recalibrate_intervals_gaussian(fitted_model, recalib_data, args):
    recalib_model = fit_isotonic(fitted_model, recalib_data)
    if recalib_model is None:
        return {}

    coverage_dict = {}
    for alpha in args.alphas:
        lower_p = [alpha/2]
        upper_p = [1 - alpha/2]
        recalib_cov = recalib_model.predict(upper_p) - recalib_model.predict(lower_p)
        logging.info("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, recalib_cov)
        coverage_dict[alpha] = {"mean": recalib_cov[0]}
    return coverage_dict

def fit_class_recalib(fitted_model, recalib_data):
    """
    @return LogisticRegression object for recalibration of classification probabilities
    """
    accept_probs = fitted_model.get_accept_prob(recalib_data.x).flatten()
    if np.mean(np.isfinite(accept_probs)) < 0.5:
        return None

    # For every x, use fitted model to get predicted probabilities
    if fitted_model.density_parametric_form == "bernoulli":
        pred_probs = fitted_model.get_prediction_probs(recalib_data.x).flatten()
        pred_probs = pred_probs.reshape(-1, 1)
        recalib_y = recalib_data.y.flatten()
    else:
        pred_probs = fitted_model.get_prediction_probs(recalib_data.x)
        recalib_y = np.array([np.where(arr == 1) for arr in recalib_data.y]).flatten()

    # Fit logistic regression, with predicted probabilities as features, and true
    # Y in recalibration set as outcome (model Y | p-hat)
    lr = LogisticRegression(C=1e5, solver='lbfgs', multi_class='auto', max_iter=4000)
    lr.fit(pred_probs, recalib_y, sample_weight=accept_probs)

    return lr

def recalibrate_intervals_bernoulli(fitted_model, recalib_data, args):
    # TODO: maybe switch to half of recalib data?
    recalib_model = fit_class_recalib(fitted_model, recalib_data)
    if recalib_model is None:
        return {}

    pred_probs = fitted_model.get_prediction_probs(recalib_data.x)
    accept_prob = fitted_model.get_accept_prob(recalib_data.x).flatten()
    # Recalibrate probabilities predicted by NN by applying model (Y | p-hat)
    pred_probs_recalib = recalib_model.predict_proba(pred_probs)

    coverage_dict = {}
    for alpha in args.alphas:
        pred_ints = fitted_model.get_prediction_interval(recalib_data.x, alpha)
        pred_ints[:,0] = pred_ints[:,0] == 0
        pred_ints[:,1] = pred_ints[:,1] == 0
        interval_element_probs = np.multiply(pred_ints, pred_probs_recalib)
        interval_prob = np.sum(interval_element_probs, axis = 1).flatten()
        # Calculate the recalibration probability weighted by aceptance prob
        recalib_cov = np.sum(interval_prob * accept_prob)/np.sum(accept_prob)
        logging.info("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, recalib_cov)
        print("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, recalib_cov)
        coverage_dict[alpha] = {"mean": recalib_cov}
    return coverage_dict

def recalibrate_intervals_multinomial(fitted_model, recalib_data, args):
    # TODO: maybe switch to half of recalib data?
    recalib_model = fit_class_recalib(fitted_model, recalib_data)
    if recalib_model is None:
        return None

    pred_probs = fitted_model.get_prediction_probs(recalib_data.x)
    accept_prob = fitted_model.get_accept_prob(recalib_data.x).flatten()
    # Recalibrate probabilities predicted by NN by applying model (Y | p-hat)
    pred_probs_recalib = recalib_model.predict_proba(pred_probs)
    coverage_dict = {}
    for alpha in args.alphas:
        pred_ints = fitted_model.get_prediction_interval(recalib_data.x, alpha)
        interval_element_probs = np.multiply(pred_ints, pred_probs_recalib)
        interval_prob = np.sum(interval_element_probs, axis = 1).flatten()
        # Calculate the recalibration probability weighted by aceptance prob
        recalib_cov = np.sum(interval_prob * accept_prob)/np.sum(accept_prob)
        logging.info("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, recalib_cov)
        print("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, recalib_cov)
        coverage_dict[alpha] = {"mean": recalib_cov}
    return coverage_dict

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)

    # Read all data
    data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    recalib_data = data_dict["train"].subset(split_dict["recalibrate_idxs"])

    # Load model
    fitted_model = load_model(args.fitted_file)
    family = fitted_model.density_parametric_form

    if family == "gaussian":
        coverage_dict = recalibrate_intervals_gaussian(fitted_model, recalib_data, args)
    elif family == "bernoulli":
        coverage_dict = recalibrate_intervals_bernoulli(fitted_model, recalib_data, args)
    elif "multinomial" in family:
        coverage_dict = recalibrate_intervals_multinomial(fitted_model, recalib_data, args)
    else:
        raise ValueError("dunno what is going on")
    print(coverage_dict)

    pickle_to_file(coverage_dict, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
