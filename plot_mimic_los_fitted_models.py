"""
Plot mimic-related stuff for length of stay
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
        default="_output/examples.png")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    return args

def plot_random_individuals(aggregator, dataset, args, min_days=2, max_days=15, bar_height=1.5, example_buffer=1.5, num_ticks=3, fontsize=14):
    agg_predictions = aggregator.aggregate_prediction_intervals(dataset.x)

    plt.clf()
    fig, ax = plt.subplots(1, figsize=(4, 1.25 * args.num_examples), sharex=True, sharey=True)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    example_height = bar_height + example_buffer

    rand_idxs = np.sort(np.random.choice(np.arange(dataset.num_obs), size=args.num_examples, replace=False))
    for example_idx in range(args.num_examples):
        idx = rand_idxs[example_idx]
        print(example_idx, idx)

        example_pred = agg_predictions[:,idx,:]
        true_label = dataset.y[idx]
        mean_accept_prob = np.mean(example_pred[:,0])
        logging.info("patient %d", idx)
        print("accept prob", mean_accept_prob)
        logging.info("  accept prob %.2f", mean_accept_prob)
        num_models = example_pred.shape[0]

        plt.text(
                2,
                example_height * (example_idx + 1) - 0.7 * example_buffer,
                "Example %d: %.02f%% accept" % (args.num_examples - example_idx, 100 * mean_accept_prob),
                fontsize=fontsize)
        prediction_intervals = []
        for model_idx in range(num_models):
            pred_range = np.exp(example_pred[model_idx, 1:])/24
            accept_prob = example_pred[model_idx,0]
            logging.info("  model %d (%.2f): %s", model_idx, accept_prob, pred_range)

            rect = Rectangle(
                    (pred_range[0], example_height * example_idx),
                    pred_range[1] - pred_range[0],
                    height=bar_height,
                    alpha=accept_prob/num_models,
                    facecolor='b')
            prediction_intervals.append(rect)
            ax.add_artist(rect)

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(True)
    plt.xticks(range(min_days, max_days, int((max_days - min_days)/num_ticks)), fontsize=fontsize)
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:d}'.format))
    plt.tick_params(axis='both', which='both', bottom='on', top='off',
        labelbottom='on', left='off', right='off', labelleft='off')
    plt.xlim(0, max_days)
    plt.xlabel("Days", fontsize=fontsize)
    plt.ylim(-1, example_height * args.num_examples + 1)
    plt.tight_layout()
    plt.savefig(args.out_example_plot)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_dataset = pickle_from_file(args.test_data_file)

    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]

    aggregator = DecisionIntervalAggregator(fitted_models, args.alpha, None)
    plot_random_individuals(aggregator, test_dataset, args)
    #plot_local_coverages(aggregator, test_dataset, args)
    plot_accept_prob_vs_age(fitted_models, test_dataset, args)
    plot_prediction_vs_age(
            fitted_models,
            test_dataset,
            args,
            "Predicted length of stay (days)",
            pred_func=lambda x: np.exp(x)/24,
            max_predict_plot=True)

if __name__ == "__main__":
    main(sys.argv[1:])
