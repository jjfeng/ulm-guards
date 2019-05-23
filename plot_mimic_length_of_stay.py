"""
visualize quantities for mimic length of stay task, with respect to average acceptance probability
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List

from outlier_density_nn import OutlierDensityNN
from accept_all_density_nn import AcceptAllDensityNN
from decision_density_nn import SimultaneousDensityDecisionNNs
from decision_prediction_base_model import DecisionPredictionBaseModel
from dataset import Dataset
from common import *
from plot_common import *

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',
        type=int,
        default=0,
        help="random seed")
    parser.add_argument('--alpha',
        type=float,
        default=0.1,
        help="pi coverage alpha")
    parser.add_argument('--result-files',
        type=str,
        default="_output/agg_result.pkl",
        help="comma separated")
    parser.add_argument('--test-data-file',
        type=str,
        default="_output/test_data.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/plot_log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.result_files = process_params(args.result_files, str)
    return args

def extract_row(accept_prob: float, val: float, key: str, type_str: str):
    try:
        return [{
            "accept_prob": accept_prob,
            "key": key,
            "value": float(val),
            "type": type_str}]
    except TypeError:
        print(val, key, type_str)
        print("failed casting to float")

def extract_rows(cost_decline: float, val_list: List[float], key: str, type_str: str):
    return [{
        "accept_prob": accept_prob,
        "key": key,
        "value": v,
        "type": type_str} for v in val_list]

def plot_results(
        results: List,
        test_data: Dataset,
        args):
    all_data = []
    # Compare recalibration methods
    for res_dict in results:
        for model_idx in range(len(res_dict["loss"])):
            accept_prob = res_dict["accept_probs"][model_idx]
            recalib_inf_dict = res_dict["recalib_indiv_coverage"][args.alpha][model_idx]
            recalib_ci = get_normal_ci(recalib_inf_dict)
            all_data += extract_row(accept_prob, recalib_ci[0], "coverage", "recalib_us")
            all_data += extract_row(accept_prob, recalib_ci[1], "coverage", "recalib_us")
            all_data += extract_row(accept_prob, res_dict["test_indiv_coverage"][args.alpha][model_idx], "coverage", "test_indiv")
            if "recalib_kuleshov_indiv_coverage" in res_dict:
                all_data += extract_row(accept_prob, res_dict["recalib_kuleshov_indiv_coverage"][args.alpha][model_idx]['mean'], "coverage", "kuleshov")
        recalib_agg_ci = res_dict["recalib_agg_coverage"][args.alpha]["ci"]
        mean_accept_prob = np.mean(res_dict["accept_probs"])
        all_data += extract_row(accept_prob, recalib_agg_ci[0], "coverage", "recalib_agg")
        all_data += extract_row(accept_prob, recalib_agg_ci[1], "coverage", "recalib_agg")
        all_data += extract_row(accept_prob, res_dict["test_agg_coverage"][args.alpha], "coverage", "test_agg")

    data_df = pd.DataFrame(all_data)
    logging.info(data_df.to_latex())

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_dataset = pickle_from_file(args.test_data_file)
    results = [pickle_from_file(res_file) for res_file in args.result_files]
    for res in results:
        res["fitted_models"] = []
        for fitted_dict in res["fitted_dicts"]:
            fitted_model = load_model_from_dict(fitted_dict)
            cost_decline = fitted_model.cost_decline
            res["fitted_models"].append(fitted_model)
    plot_results(results, test_dataset, args)
    print("DONE PLOTTING MIMIC")

if __name__ == "__main__":
    main(sys.argv[1:])
