import os
import time
import numpy as np
from numpy import ndarray
import scipy.stats
import pickle
import json
from itertools import chain, combinations
from operator import mul
from functools import reduce

IMG_EXTENSIONS = ["jpeg", "png", "jpg", "TIF", "tif"]
THRES = 1e-7
ALMOST_ZERO = 0

def get_normal_ci(mean_se_dict, ci_alpha=0.05, min_lower=None, max_upper=None):
    """
    @param ci_alpha: produce CI with 1 - ci_alpha coverage
    @param min_lower: minimum value for lower bound, if it exists
    @param max_upper: maximum value for upper bound, if it exists

    @return (lower, upper) for CI with desired coverage
    """
    mean = mean_se_dict["mean"]
    se = mean_se_dict["se"]
    z_factor = scipy.stats.norm.ppf(1 - ci_alpha/2)
    ci_lower = mean - z_factor * se
    ci_upper = mean + z_factor * se
    if min_lower is not None:
        ci_lower = max(min_lower, ci_lower)
    if max_upper is not None:
        ci_upper = min(max_upper, ci_upper)
    return (ci_lower, ci_upper)

def get_interval_width(PIs: ndarray):
    if PIs.shape[1] == 2:
        # Binomial or contiuous
        return PIs[:,1] - PIs[:,0]
    elif PIs.shape[1] > 2:
        # multinomial
        return np.sum(PIs, axis=1)
    else:
        raise ValueError("weird. what is this?")

def distance_from_interval(PIs: ndarray, y: ndarray):
    """
    @param PIs: first col is lower PI bound, second col is upper PI bound
    @param y: the y we are testing if in interval
    @param jiggle: a jiggling parameter in case we are comparing values that are very close

    @return binary value whether y is in PI
    """
    if len(y.shape) == 1 or y.shape[1] == 1:
        lower_dist = np.maximum(PIs[:,0] - y.flatten(), 0)
        upper_dist = np.maximum(y.flatten() - PIs[:,1], 0)
        return lower_dist + upper_dist
    else:
        raise ValueError("not allowed for multinomial")

def is_within_interval(PIs: ndarray, y: ndarray, jiggle: float = 1e-10):
    """
    @param PIs: first col is lower PI bound, second col is upper PI bound
    @param y: the y we are testing if in interval
    @param jiggle: a jiggling parameter in case we are comparing values that are very close

    @return binary value whether y is in PI
    """
    if len(y.shape) == 1 or y.shape[1] == 1:
        # Binomial or contiuous
        lower_check = PIs[:,0] - jiggle <= y.flatten()
        upper_check = PIs[:,1] + jiggle >= y.flatten()
        within_check = lower_check * upper_check
        return within_check
    else:
        # multinomial
        upper_violation = y > PIs + jiggle
        within_check = ~np.any(upper_violation, axis=1)
        return within_check

def pickle_to_file(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, protocol=-1)

def pickle_from_file(file_name):
    with open(file_name, "rb") as f:
        out = pickle.load(f)
    return out

def read_list_file(list_file):
    """
    read a bunch of lines in a file
    """
    with open(list_file, "r") as f:
        lines = list(map(lambda x: x.replace("\n", ""), f.readlines()))
    return lines

def process_params(param_str, dtype, split_str=","):
    if param_str:
        return [dtype(r) for r in param_str.split(split_str)]
    else:
        return []

def load_model(fitted_file):
    """
    @param fitted_file: file name
    Load a decision prediction model
    @return the fitted model
    """
    print("opening file", fitted_file)
    try:
        fitted_dict = pickle_from_file(fitted_file)
    except FileNotFoundError as e:
        print(e)
        return None
    return load_model_from_dict(fitted_dict)

def load_model_from_dict(fitted_dict):
    """
    Load a decision prediction model
    @return the fitted model
    """
    nn_class = fitted_dict["nn_class"]
    fitted_model = nn_class(**fitted_dict["hyperparams"])
    fitted_model._init_nn(init_inner=True)
    fitted_model.set_model_params(fitted_dict["fitted_params"])
    #print(fitted_dict["fitted_params"])
    return fitted_model

def get_multinomial_entropy(probs):
    return -np.sum(probs * np.log(probs), axis=1)

def get_normal_dist_entropy(sigma):
    return 0.5 * np.log(np.power(sigma, 2) * 2 * np.pi * np.e)

def get_mu_sigma_of_mixture_normals(all_mus, all_sigmas):
    final_mu = np.mean(np.concatenate(all_mus, axis=1), axis=1, keepdims=True)
    final_sigma = np.sqrt(
            np.mean(np.concatenate(np.power(all_sigmas, 2), axis=1), axis=1, keepdims=True)
            + np.mean(np.concatenate(np.power(all_mus, 2), axis=1), axis=1, keepdims=True)
            - np.power(final_mu, 2))
    return final_mu, final_sigma

def get_bernoulli_pred_interval(mu, alpha):
    pred_set_is_one = mu >= 1 - alpha
    pred_set_is_zero = mu <= alpha
    lower_pi = pred_set_is_one
    upper_pi = 1 - pred_set_is_zero
    return np.concatenate([lower_pi, upper_pi], axis=1)

def get_multinomial_pred_interval(predicted_mu, alpha):
    all_prediction_intervals = []
    # Used to track how much remaining prob uncovered when we try to cover 1-alpha
    #all_remaining_prob = []
    for i in range(predicted_mu.shape[0]):
        mu = predicted_mu[i,:]
        prediction_interval = np.zeros((1, mu.size))
        sorted_idxs = np.argsort(mu)
        curr_idx = 0
        tot_alpha = 0
        while tot_alpha < alpha:
            mu_idx = mu[sorted_idxs[curr_idx]]
            if tot_alpha + mu_idx > alpha:
                break
            tot_alpha += mu_idx
            curr_idx += 1
        prediction_interval[0, sorted_idxs[curr_idx:]] = 1
        all_prediction_intervals.append(prediction_interval)

        #all_remaining_prob.append(tot_alpha)

    #print("avg remaining prob", np.mean(all_remaining_prob))
    return np.concatenate(all_prediction_intervals, axis=0)

def is_integer(x: ndarray):
    return np.equal(np.mod(x, 1), 0)

def mult_list(x: list):
    return reduce(mul, x, 1)

def make_scratch_dir(args):
    scratch_dir = args.scratch_dir
    if args.do_distributed and args.num_ensemble > 1:
        while True:
            scratch_dir = "%s/%d" % (args.scratch_dir, np.random.randint(1) + int(time.time()))
            try:
                os.mkdir(scratch_dir)
            except OSError:
                print ("Creation of the directory %s failed" % scratch_dir)
                time.sleep(1)
            else:
                print ("Successfully created the directory %s " % scratch_dir)
                break
    return scratch_dir
