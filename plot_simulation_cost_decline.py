"""
visualize how validation loss and acceptance probability
change with different delta values (delta = decline cost)
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from plot_fitted import eval_coverage
from common import load_model, pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci, get_normal_dist_entropy

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--num-test',
        type=int,
        default=2000)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        default="_output/data_split.pkl")
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl",
        help="comma separated")
    parser.add_argument('--coverage-files',
        type=str,
        default="_output/recalibrated_coverages.pkl",
        help="comma separated")
    parser.add_argument('--plot-diam-file',
        type=str,
        default="_output/delta_diam.png")
    parser.add_argument('--plot-coverage-file',
        type=str,
        default="_output/delta_coverage.png")
    parser.add_argument('--plot-accept-file',
        type=str,
        default="_output/delta_accept.png")
    parser.add_argument('--plot-accept-region-file',
        type=str,
        default="_output/delta_accept_region.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    args.coverage_files = process_params(args.coverage_files, str)
    return args

def plot_accepted_rejected_region(data_dict, fitted_models, args, mesh_size=0.05):
    """
    Plot acceptance region. Last row plot entropy
    """
    COLORS = ['orange', 'red']
    LINESTYLES = ['dashed', 'dotted']

    num_models = len(fitted_models)
    print("num models", num_models)
    # Look at the region we accepted
    mesh_coords, (xx, yy) = data_dict["support_sim_settings"].generate_grid(mesh_size)
    y_given_x_sigma = data_dict["data_gen"].sigma_func(mesh_coords)
    entropy = get_normal_dist_entropy(y_given_x_sigma)

    all_accept_probs = []
    for fitted_model in fitted_models:
        print(fitted_model.cost_decline)
        x_accept_probs = fitted_model.get_accept_prob(mesh_coords)
        all_accept_probs.append(x_accept_probs)

    fig, ax = plt.subplots(nrows=1, figsize=(4,4))
    for idx, x_accept_probs in enumerate(all_accept_probs):
        cs = ax.contour(
                xx,
                yy,
                x_accept_probs.reshape(xx.shape),
                levels=[0.99],
                colors=COLORS[idx],
                linestyles=LINESTYLES[idx],
                linewidths=4)
    cs = ax.contourf(xx, yy, entropy.reshape(xx.shape), cmap='gray')
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(args.plot_accept_region_file)

def plot_PI_diam(fitted_dicts, data, args):
    """
    Plot diameter of prediction intervals
    """
    diam_data = []
    for fitted_dict in fitted_dicts:
        fitted_model = fitted_dict["model"]
        coverage_dict = fitted_dict["coverage_dict"]
        accept_probs = fitted_model.get_accept_prob(data.x).flatten()
        accept_mask = accept_probs > 0.3
        cost_decline = fitted_model.cost_decline
        for alpha in coverage_dict.keys():
            alpha_PIs = fitted_model.get_prediction_interval(data.x, alpha)
            diameter_PIs = alpha_PIs[:,1] - alpha_PIs[:,0]
            avg_accept_diam = np.mean(diameter_PIs[accept_mask])
            #avg_accept_diam = np.max(diameter_PIs[accept_probs])
            diam_row = {
                    "cost_decline": cost_decline,
                    "alpha": alpha,
                    "diam_PI": avg_accept_diam}
            diam_data.append(diam_row)

    diam_df = pd.DataFrame(diam_data)

    plt.clf()
    sns.relplot(
            x="cost_decline",
            y="diam_PI",
            col="alpha",
            data=diam_df)
            #data=diam_df)
    plt.savefig(args.plot_diam_file)

def plot_coverages(fitted_dicts, test_data, args):
    coverage_data = []
    for fitted_dict in fitted_dicts:
        fitted_model = fitted_dict["model"]
        coverage_dict = fitted_dict["coverage_dict"]
        cost_decline = fitted_model.cost_decline
        for alpha, inference_dict in coverage_dict.items():
            actual_coverage = eval_coverage(fitted_model, test_data, alpha)
            coverage_row = {
                    "cost_decline": cost_decline,
                    "alpha": alpha,
                    "type": "actual",
                    "coverage": actual_coverage}
            coverage_data.append(coverage_row)
            coverage_row = {
                    "cost_decline": cost_decline,
                    "alpha": alpha,
                    "type": "recalib",
                    "coverage": inference_dict["cov_given_accept"]["mean"]}
            coverage_data.append(coverage_row)
            coverage_row = {
                    "cost_decline": cost_decline,
                    "alpha": alpha,
                    "type": "nominal",
                    "coverage": 1 - 2 * alpha}
            coverage_data.append(coverage_row)

    coverage_df = pd.DataFrame(coverage_data)
    #print(coverage_df)

    plt.clf()
    sns.relplot(
            x="cost_decline",
            y="coverage",
            col="alpha",
            hue="type",
            data=coverage_df)
            #data=coverage_df)
    plt.savefig(args.plot_coverage_file)

def plot_accept_probs(
        fitted_models,
        data,
        args):
    all_accept_probs = []
    cost_declines = []
    for fitted_model in fitted_models:
        accept_probs = fitted_model.get_accept_prob(data.x)
        all_accept_probs.append(np.mean(accept_probs))
        cost_declines.append(fitted_model.cost_decline)

    print("accept probs", all_accept_probs)
    plt.clf()
    sns.regplot(
            cost_declines,
            all_accept_probs,
            fit_reg=False)
    plt.savefig(args.plot_accept_file)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    # Read all data
    orig_data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    recalib_data = orig_data_dict["train"].subset(split_dict["recalibrate_idxs"])
    args.num_p = recalib_data.x.shape[1]

    # Load models
    fitted_dicts = []
    #for fitted_file, coverage_file in zip(args.fitted_files, args.coverage_files):
    for fitted_file in args.fitted_files:
        fitted_model = load_model(fitted_file)

        #coverage_dict = pickle_from_file(coverage_file)

        fitted_dicts.append({
            "model": fitted_model})
            #"coverage_dict": coverage_dict})
    print("fitted dicts", len(fitted_dicts))

    # Do all the plotting
    new_data, _ = orig_data_dict["data_gen"].create_data(args.num_test)
    #plot_PI_diam(fitted_dicts, new_data, args)
    #plot_coverages(fitted_dicts, new_data, args)
    #plot_accept_probs(
    #        [d["model"] for d in fitted_dicts],
    #        new_data,
    #        args)
    if args.num_p == 2:
        plot_accepted_rejected_region(
            orig_data_dict,
            [d["model"] for d in fitted_dicts],
            args)

if __name__ == "__main__":
    main(sys.argv[1:])
