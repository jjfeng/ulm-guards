"""
Assess aggregate models
"""
import sys
import argparse
import logging
import numpy as np

from decision_interval_indpt_aggregator import DecisionIntervalIndptAggregator
from decision_interval_recalibrator import DecisionIntervalRecalibrator
from decision_density_nn import SimultaneousDensityDecisionNNs
from dataset import Dataset
from common import *
from plot_common import *


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--test-data-file',
        type=str,
        default="_output/test_data.pkl")
    parser.add_argument('--alpha',
        type=float,
        default=0.1)
    parser.add_argument('--ci-alpha',
        type=float,
        default=0.05)
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--coverage-files',
        type=str,
        default="_output/recalibrated_coverages.pkl")
    parser.add_argument('--coverage-kuleshov-files',
        type=str,
        default=None)
    parser.add_argument('--out-file',
        type=str,
        default="_output/aggregate_results.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log_agg.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    args.coverage_files = process_params(args.coverage_files, str)
    args.coverage_kuleshov_files = process_params(args.coverage_kuleshov_files, str)
    return args

def get_prob_accept(fitted_model: SimultaneousDensityDecisionNNs, dataset: Dataset, eps : float = 1e-30):
    accept_prob = fitted_model.get_accept_prob(dataset.x).flatten()
    return np.mean(accept_prob)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(args)

    test_dataset = pickle_from_file(args.test_data_file)

    # Remove any really bad modles
    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]
    test_accept_probs = [
            get_prob_accept(fitted_model, test_dataset)
            for fitted_model in fitted_models]
    print("good model indices", np.where(np.isfinite(test_accept_probs))[0])
    good_idxs = np.where(np.isfinite(test_accept_probs))[0]
    fitted_models = [fitted_models[idx] for idx in good_idxs]
    assert(len(fitted_models) > 0)
    cost_decline = fitted_models[0].cost_decline

    fitted_dicts = []
    agg_dict = {}
    for good_idx in good_idxs:
        fitted_file = args.fitted_files[good_idx]
        coverage_file = args.coverage_files[good_idx]
        fitted_dict = pickle_from_file(fitted_file)
        fitted_dicts.append(fitted_dict)

        coverage_dict = pickle_from_file(coverage_file)
        for alpha, inference_dict in coverage_dict.items():
            if alpha not in agg_dict:
                agg_dict[alpha] = []
            agg_dict[alpha].append(inference_dict)

    test_accept_probs = [
            get_prob_accept(fitted_model, test_dataset)
            for fitted_model in fitted_models]
    print("accept prob", test_accept_probs)
    test_log_liks = None
    if fitted_models[0].has_density:
        test_loss = [
            get_log_lik_given_accept(fitted_model, test_dataset)
            for fitted_model in fitted_models]
    else:
        test_loss = [
            get_interval_loss_given_accept(fitted_model, test_dataset, args.alpha)
            for fitted_model in fitted_models]

    test_coverage_agg_results = {}
    recalib_coverage_agg_results = {}
    test_coverage_indiv_results = {}
    recalib_coverage_indiv_results = {}
    pi_width_agg_results = {}
    for alpha, inference_dicts in agg_dict.items():
        recalib_coverage_indiv_results[alpha] = [
                inf_dict["cov_given_accept"] for inf_dict in inference_dicts]

        recalibrators = [
                DecisionIntervalRecalibrator(fitted_model, alpha) for fitted_model in fitted_models]
        test_coverage_indiv_results[alpha] = [
                recalibrator.eval_cov_given_accept(test_dataset) for recalibrator in recalibrators]
        for fold_idx, recalibrator in enumerate(recalibrators):
            test_inf_dict = recalibrator.recalibrate(test_dataset)["cov_given_accept"]
            test_ci = get_normal_ci(test_inf_dict, args.ci_alpha)
            logging.info("Test coverage %d: mean: %.3f, ci: %.3f, %.3f" % (
                fold_idx, test_inf_dict["mean"], test_ci[0], test_ci[1]))

        aggregator = DecisionIntervalIndptAggregator(fitted_models, alpha, inference_dicts)
        agg_cov_given_accept_dict = aggregator.calc_agg_cover_given_accept(args.ci_alpha)
        recalib_coverage_agg_results[alpha] = agg_cov_given_accept_dict

        agg_inf_dict = aggregator.eval_cov_given_accept(test_dataset)
        logging.info("Test agg coverage: mean: %.3f, se: %.3f" % (
            agg_inf_dict["cov_given_accept"]["mean"], agg_inf_dict["cov_given_accept"]["se"]))
        test_coverage_agg_results[alpha] = agg_inf_dict["cov_given_accept"]["mean"]
        pi_width_agg_results[alpha] = [
            get_pi_width_given_accept(fitted_model, test_dataset, alpha)
            for fitted_model in fitted_models]
    logging.info("our method indiv %s", recalib_coverage_indiv_results)
    logging.info("actual indiv coverage %s", test_coverage_indiv_results)

    if args.coverage_kuleshov_files:
        kuleshov_recalib_coverage_indiv_results = {}
        for good_idx in good_idxs:
            coverage_kuleshov_file = args.coverage_kuleshov_files[good_idx]
            coverage_kuleshov_dict = pickle_from_file(coverage_kuleshov_file)
            for alpha, val_dict in coverage_kuleshov_dict.items():
                if alpha not in kuleshov_recalib_coverage_indiv_results:
                    kuleshov_recalib_coverage_indiv_results[alpha] = []
                kuleshov_recalib_coverage_indiv_results[alpha].append(val_dict)

        logging.info("kuleshov indiv %s", kuleshov_recalib_coverage_indiv_results)

    agg_results = {
            "loss": test_loss,
            "cost_decline": cost_decline,
            "accept_probs": test_accept_probs,
            "test_agg_coverage": test_coverage_agg_results,
            "recalib_agg_coverage": recalib_coverage_agg_results,
            "test_indiv_coverage": test_coverage_indiv_results,
            "recalib_indiv_coverage": recalib_coverage_indiv_results,
            "pi_widths": pi_width_agg_results,
            "fitted_dicts": fitted_dicts,
    }
    if args.coverage_kuleshov_files:
        agg_results["recalib_kuleshov_indiv_coverage"] = kuleshov_recalib_coverage_indiv_results
    logging.info(agg_results)
    pickle_to_file(agg_results, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
