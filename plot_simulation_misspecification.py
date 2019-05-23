"""
visualize where we accept (under model misspecification)
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
    parser.add_argument('--plot-accept-region-file',
        type=str,
        default="_output/delta_accept_region.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    return args

def plot_accepted_rejected_region(data_dict, fitted_models, args, mesh_size=0.2):
    """
    Plot acceptance region
    """
    num_models = len(fitted_models)
    # Look at the region we accepted
    mesh_coords, (xx, yy) = data_dict["support_sim_settings"].generate_grid(mesh_size)
    y_given_x_sigma = data_dict["data_gen"].sigma_func(mesh_coords)
    entropy = get_normal_dist_entropy(y_given_x_sigma)

    all_accept_probs = []
    for fitted_model in fitted_models:
        x_accept_probs = fitted_model.get_accept_prob(mesh_coords)
        all_accept_probs.append(x_accept_probs)

    plt.clf()
    fig, ax = plt.subplots(nrows=1, figsize=(4,4))
    for idx, x_accept_probs in enumerate(all_accept_probs):
        cs = ax.contourf(xx, yy, x_accept_probs.reshape(xx.shape))
        cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(args.plot_accept_region_file)

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
    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]

    # Do all the plotting
    new_data, _ = orig_data_dict["data_gen"].create_data(args.num_test)
    if args.num_p == 2:
        plot_accepted_rejected_region(
            orig_data_dict,
            fitted_models,
            args)

if __name__ == "__main__":
    main(sys.argv[1:])
