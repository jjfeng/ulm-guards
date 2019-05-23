"""
Assess aggregate intervals
"""
import sys
import argparse
import logging
import numpy as np
import scipy.stats

from common import pickle_to_file, pickle_from_file, process_params, get_normal_ci, load_model
from decision_interval_aggregator import DecisionIntervalAggregator
from decision_interval_indpt_aggregator import DecisionIntervalIndptAggregator
from decision_interval_recalibrator import DecisionIntervalRecalibrator
from plot_common import *


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--num-rand',
        type=int,
        default=20)
    parser.add_argument('--num-test',
        type=int,
        default=6000)
    parser.add_argument('--ci-alpha',
        type=float,
        default=0.1)
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--coverage-files',
        type=str,
        default="_output/recalibrated_coverages.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/aggregate_coverages.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log_agg.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    args.coverage_files = process_params(args.coverage_files, str)
    return args

def get_individual_ci_diams(inference_dicts, ci_alpha):
    """
    Calculate the width of the CIs from the single training validation split
    """
    ci_diams = []
    for inf_dict in inference_dicts:
        individual_ci = get_normal_ci(inf_dict["cov_given_accept"], ci_alpha, min_lower=0, max_upper=1)
        ci_diams.append(individual_ci[1] - individual_ci[0])
    return ci_diams

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(args)

    data_dict = pickle_from_file(args.data_file)
    test_data, _ = data_dict["data_gen"].create_data(args.num_test)
    fitted_models = []
    agg_dict = {}
    for fitted_file, coverage_file in zip(args.fitted_files, args.coverage_files):
        fitted_model = load_model(fitted_file)
        fitted_models.append(fitted_model)
        coverage_dict = pickle_from_file(coverage_file)
        for pi_alpha, inference_dict in coverage_dict.items():
            if pi_alpha not in agg_dict:
                agg_dict[pi_alpha] = []
            agg_dict[pi_alpha].append(inference_dict)

    unif_x = data_dict["support_sim_settings"].support_unif_rvs(args.num_test)
    unif_test_data = data_dict["data_gen"].create_data_given_x(unif_x)

    coverage_agg_results = {}
    for pi_alpha, inference_dicts in agg_dict.items():
        aggregator = DecisionIntervalAggregator(fitted_models, pi_alpha, inference_dicts)
        indiv_test_datas = [
                data_dict["data_gen"].create_data(args.num_test)[0]
                for _ in fitted_models]
        indiv_test_inf_dicts = [
                DecisionIntervalRecalibrator(fitted_model, pi_alpha).recalibrate(indiv_test_data)
                for fitted_model, indiv_test_data in zip(fitted_models, indiv_test_datas)]
        individual_is_covereds = []
        for test_coverage_dict, inf_dict in zip(indiv_test_inf_dicts, inference_dicts):
            print(inf_dict)
            test_coverage = test_coverage_dict["cov_given_accept"]["mean"]
            test_coverage_ci = get_normal_ci(test_coverage_dict["cov_given_accept"], args.ci_alpha)
            individual_ci = get_normal_ci(inf_dict["cov_given_accept"], args.ci_alpha)
            indiv_covered = individual_ci[0] <= test_coverage and test_coverage <= individual_ci[1]
            logging.info("indiv est %f ci %s", inf_dict["cov_given_accept"]["mean"], individual_ci)
            logging.info("true indiv %f ci %s", test_coverage, test_coverage_ci)
            logging.info("indiv is covered? %s", indiv_covered)
            individual_is_covereds.append(indiv_covered)

        # Calculate the width of the individual CI diams for comparison
        individual_ci_diams = get_individual_ci_diams(inference_dicts, args.ci_alpha)

        # Evaluate if the true coverage value is covered
        agg_cov_given_accept_dict = aggregator.calc_agg_cover_given_accept(args.ci_alpha)
        true_cov_given_accept_dict = aggregator.eval_cov_given_accept(test_data)["cov_given_accept"]
        true_cov_given_accept = true_cov_given_accept_dict["mean"]
        agg_ci = agg_cov_given_accept_dict["ci"]
        is_covered = true_cov_given_accept > agg_ci[0] and true_cov_given_accept < agg_ci[1]

        # Evaluate coverage if using independence assumption
        indpt_aggregator = DecisionIntervalIndptAggregator(fitted_models, pi_alpha, inference_dicts)
        indpt_agg_cov_given_accept_dict = indpt_aggregator.calc_agg_cover_given_accept(args.ci_alpha)
        indpt_ci = indpt_agg_cov_given_accept_dict["ci"]
        indpt_is_covered = true_cov_given_accept > indpt_ci[0] and true_cov_given_accept < indpt_ci[1]

        coverage_agg_results[pi_alpha] = {
                "is_covered": {
                    "agg": [is_covered],
                    "independent": [indpt_is_covered],
                    "individual": individual_is_covereds},
                "ci_diams": {
                    "agg": [agg_ci[1] - agg_ci[0]],
                    "independent": [indpt_ci[1] - indpt_ci[0]],
                    "individual": individual_ci_diams},
                "true_cov": {
                    "agg": [true_cov_given_accept],
                    "independent": [true_cov_given_accept],
                    "individual": [
                        test_inf_dict["cov_given_accept"]["mean"]
                        for test_inf_dict in indiv_test_inf_dicts]}
        }

        # Evaluate local coverage
        local_coverages = assess_local_agg_coverage_true(
                aggregator,
                test_data,
                data_dict["data_gen"])
        for key, val in local_coverages.items():
            coverage_agg_results[pi_alpha][key] = val

        logging.info("PI alpha %f", pi_alpha)
        logging.info("estimated agg cover given accept %f %s", agg_cov_given_accept_dict["mean"], agg_ci)
        logging.info("indepttt estimated agg cover given accept %f %s", indpt_agg_cov_given_accept_dict["mean"], indpt_ci)
        logging.info("true cov given accept %f, se %f", true_cov_given_accept, true_cov_given_accept_dict["se"])
        logging.info("is  covered? %s", is_covered)
        logging.info("indept is  covered? %s", indpt_is_covered)

    logging.info(coverage_agg_results)
    pickle_to_file(coverage_agg_results, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
