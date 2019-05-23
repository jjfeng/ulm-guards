"""
visualize some stuff that we fit already
"""
import sys
import argparse
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from common import pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci, load_model


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=40)
    parser.add_argument('--num-test',
        type=int,
        default=2000)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--recalibrated-file',
        type=str,
        default="_output/recalibrated.pkl")
    parser.add_argument('--plot-decision-file',
        type=str,
        default="_output/decision.png")
    parser.add_argument('--plot-density-accept-file',
        type=str,
        default="_output/density_accept.png")
    parser.add_argument('--plot-density-decline-file',
        type=str,
        default="_output/density_decline.png")
    parser.set_defaults()
    args = parser.parse_args()
    return args

def plot_accepted_rejected_region(data_dict, fitted_model, args, size_rand = 2000):
    # Look at the region we accepted
    new_data, _ = data_dict["data_gen"].create_data(size_rand)
    x_random = new_data.x
    x_accept_probs = fitted_model.get_accept_prob(x_random)
    entropy = data_dict["data_gen"].entropy_func(x_random)
    is_accepted = x_accept_probs.flatten() > 0.5
    print("accepted entropy", np.median(entropy[is_accepted]))
    print("rejected entropy", np.median(entropy[~is_accepted]))
    plt.clf()
    sns.violinplot(is_accepted, entropy)
    plt.savefig(args.plot_decision_file)

def plot_densities(test_data, fitted_model, args):
    # Look at how good the density estimates are in the
    # accept vs reject region
    predict_densities = fitted_model.get_density(test_data.x, test_data.y)
    x_accept_probs = fitted_model.get_accept_prob(test_data.x)
    print(x_accept_probs)
    mask = x_accept_probs > 0.5
    print("num accept", np.sum(mask))
    print("num reject", np.sum(~mask))
    plt.clf()
    sns.regplot(
            test_data.true_pdf[mask],
            predict_densities[mask],
            scatter_kws={"s": 8, "alpha": 0.3},
            lowess=True)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(args.plot_density_accept_file)

    plt.clf()
    sns.regplot(
            test_data.true_pdf[~mask],
            predict_densities[~mask],
            scatter_kws={"s": 8, "alpha": 0.3},
            lowess=True)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(args.plot_density_decline_file)

def eval_coverage(fitted_model, test_data, alpha):
    accept_probs = fitted_model.get_accept_prob(test_data.x).flatten()
    alpha_PIs = fitted_model.get_prediction_interval(test_data.x, alpha)

    accept_mask = np.random.binomial([1] * test_data.num_obs, accept_probs).astype(bool)
    within_check = is_within_interval(alpha_PIs[accept_mask, :], test_data.y[accept_mask])
    return np.mean(within_check.flatten())

def check_recalibration_covered(fitted_model, recalibrated_dict, test_data):
    for alpha, inference_dict in recalibrated_dict.items():
        test_eval_coverage = eval_coverage(fitted_model, test_data, alpha)
        estimated_ci = get_normal_ci(inference_dict["cov_given_accept"])
        print("alpha", alpha)
        print("true coverage from test set", test_eval_coverage, "estimated ci", estimated_ci)
        print("is covered?", test_eval_coverage > estimated_ci[0] and test_eval_coverage < estimated_ci[1])


def main(args=sys.argv[1:]):
    args = parse_args(args)
    data_dict = pickle_from_file(args.data_file)
    test_data, _ = data_dict["data_gen"].create_data(args.num_test, args.seed)
    args.num_p = test_data.x.shape[1]

    fitted_model = load_model(args.fitted_file)

    # Look at the region we accepted
    plot_accepted_rejected_region(data_dict, fitted_model, args)

    # Look at how good the density estimates are in the
    # accept vs reject region
    plot_densities(test_data, fitted_model, args)

    recalibrated_dict = pickle_from_file(args.recalibrated_file)
    check_recalibration_covered(fitted_model, recalibrated_dict, test_data)

if __name__ == "__main__":
    main(sys.argv[1:])
