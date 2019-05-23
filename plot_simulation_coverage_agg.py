"""
visualize how coverage varies with num train
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

from common import pickle_from_file, process_params

METHOD_DICT = {
        "agg": "Aggregate",
        "individual": "Individual",
        "independent": "Aggregate"}
PLOT_TITLE_DICT = {
        "is_covered": "CI Coverage",
        "ci_diams": "Diameter",
        "true_cov": "True PI coverage",
        "local_coverage_var": "local cov var",
        "local_coverage_iqr": "local cov iqr"}

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--pi-alpha',
        type=float,
        default=0.1,
        help="(1-pi_alpha) prediction interval")
    parser.add_argument('--num-trains',
        type=str,
        default="1",
        help="comma separated")
    parser.add_argument('--coverage-files',
        type=str,
        default="_output/agg_coverages.pkl",
        help="comma separated")
    parser.add_argument('--plot-file',
        type=str,
        default="_output/coverage_vs_num_train.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.coverage_files = process_params(args.coverage_files, str)
    args.num_trains = process_params(args.num_trains, int)
    return args

def plot_coverage_vs_num_train(coverage_results, args):
    all_data = []
    for num_train, coverage_dict in coverage_results:
        results_dict = coverage_dict[args.pi_alpha]
        for metric_key, res_dict in results_dict.items():
            if metric_key in ["is_covered", "ci_diams", "local_coverage_iqr"]:
                for type_key, values in res_dict.items():
                    if type_key == "agg" and metric_key != "local_coverage_iqr":
                        continue
                    for val in values:
                        data_row = {
                            "num_train": num_train,
                            "value": float(val),
                            "measure": metric_key,
                            "type": METHOD_DICT[type_key]}
                        all_data.append(data_row)

    coverage_data = pd.DataFrame(all_data)
    print(coverage_data)
    is_covered_mask = coverage_data.measure == "is_covered"
    is_train = coverage_data.num_train == 2880 * 2

    method_mask = coverage_data.type == "Aggregate"
    summ = np.sum(coverage_data[is_covered_mask & method_mask & is_train].value)
    print("indpt", summ, "out of", np.sum(is_covered_mask & method_mask & is_train))

    method_mask = coverage_data.type == "Individual"
    summ = np.sum(coverage_data[is_covered_mask & method_mask & is_train].value)
    print("indiv", summ, "out of", np.sum(is_covered_mask & method_mask & is_train))

    plt.clf()
    plt.figure(figsize=(2,4))
    sns.set(font_scale=1.25, style="white")
    sns.despine()
    g = sns.relplot(
            x="num_train",
            y="value",
            hue="type",
            style="type",
            col="measure",
            kind="line",
            data=coverage_data,
            facet_kws={"sharey":False},
            ci=95)
            #ci="sd")
    g = g.set_titles("").set_xlabels("Number of training obs")
    g.axes[0,0].set_ylabel("CI Coverage")
    g.axes[0,1].set_ylabel("CI Width")
    g.axes[0,2].set_ylabel("PI Coverage IQR")
    plt.savefig(args.plot_file)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    coverage_results = []
    for coverage_file, num_train in zip(args.coverage_files, args.num_trains):
        coverage_result = pickle_from_file(coverage_file)
        coverage_results.append((num_train, coverage_result))

    plot_coverage_vs_num_train(coverage_results, args)

if __name__ == "__main__":
    main(sys.argv[1:])
