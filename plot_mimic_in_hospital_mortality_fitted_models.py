"""
Plot mimic-related stuff for in-hospital mortality
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
from matplotlib.patches import Rectangle

from decision_interval_aggregator import DecisionIntervalAggregator
from common import *
from plot_mimic_common import *


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--num-rand',
        type=int,
        help='number of random obs to sample for local coverage',
        default=400)
    parser.add_argument('--num-nearest-neighbors',
        type=int,
        help='number of nearest neighbors to assess local coverage',
        default=30)
    parser.add_argument('--num-examples',
        type=int,
        help='number of example patients to show',
        default=5)
    parser.add_argument('--alpha',
        type=float,
        help="(1-alpha) prediction interval",
        default=0.1)
    parser.add_argument('--train-data-file',
        type=str,
        default="_output/train_data.pkl")
    parser.add_argument('--test-data-file',
        type=str,
        default="_output/test_data.pkl")
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--out-age-plot',
        type=str,
        default="_output/accept_vs_age.png")
    parser.add_argument('--out-age-pred-plot',
        type=str,
        default="_output/pred_vs_age.png")
    parser.add_argument('--out-local-coverage-plot',
        type=str,
        default="_output/local_coverage.png")
    parser.add_argument('--out-example-plot',
        type=str,
        default="_output/example.png")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    return args

def plot_random_individuals(aggregator, dataset, args, square_dim=1, example_buffer=1, btw_square_buffer=1, fontsize=12):
    agg_predictions = aggregator.aggregate_prediction_intervals(dataset.x)
    num_models = aggregator.num_models

    plt.clf()
    fig, ax = plt.subplots(1, figsize=(2, args.num_examples), sharex=True, sharey=True)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    example_height = square_dim + example_buffer

    rand_idxs = np.sort(np.random.choice(np.arange(dataset.num_obs), size=args.num_examples, replace=False))
    for example_idx in range(args.num_examples):
        idx = rand_idxs[example_idx]
        example_pred = agg_predictions[:,idx,:]
        true_label = dataset.y[idx]

        accept_probs = example_pred[:,0]
        mean_accept_prob = np.mean(accept_probs)
        prob_zero = np.sum((example_pred[:,1] == 0) * accept_probs)/num_models
        prob_one = np.sum((example_pred[:,2] == 1) * accept_probs)/num_models
        logging.info("patient %d %.02f%%", idx, 100 * mean_accept_prob)
        logging.info("  %.02f%% %.02f%%", 100 * prob_zero, 100 * prob_one)

        plt.text(
                0,
                example_height * (example_idx + 1) - 0.7 * example_buffer,
                "Example %d: %.02f%%" % (args.num_examples - example_idx, 100 * mean_accept_prob),
                fontsize=fontsize)
        rect0 = Rectangle(
                (0, example_height * example_idx),
                square_dim,
                height=square_dim,
                alpha=prob_zero,
                facecolor='b')
        rect1 = Rectangle(
                (square_dim + btw_square_buffer, example_height * example_idx),
                square_dim,
                height=square_dim,
                alpha=prob_one,
                facecolor='b')
        ax.add_artist(rect0)
        ax.add_artist(rect1)

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(True)
    plt.xticks(
            [square_dim/2, square_dim + square_dim/2 + btw_square_buffer],
            ["True", "False"],
            fontsize=fontsize)
    plt.tick_params(axis='both', which='both', bottom='on', top='off',
        labelbottom='on', left='off', right='off', labelleft='off')
    plt.xlim(0, square_dim * 2 + btw_square_buffer)
    plt.xlabel("Survives ICU stay?", fontsize=fontsize)
    plt.ylim(-0.25, example_height * args.num_examples + 0.25)
    plt.tight_layout()
    plt.savefig(args.out_example_plot)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_dataset = pickle_from_file(args.test_data_file)

    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]
    aggregator = DecisionIntervalAggregator(fitted_models, args.alpha, None)
    #plot_local_coverages(aggregator, test_dataset, args)
    plot_accept_prob_vs_age(fitted_models, test_dataset, args)
    plot_prediction_vs_age(fitted_models, test_dataset, args, "Predicted mortality")

    plot_random_individuals(aggregator, test_dataset, args)

if __name__ == "__main__":
    main(sys.argv[1:])
