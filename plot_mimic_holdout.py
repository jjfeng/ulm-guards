"""
Plot mimic-related stuff for in-hospital mortality, for holdout
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd

from decision_prediction_combo import *
from common import *
from plot_mimic_common import *
from plot_common import *


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--cost-decline',
        type=float,
        default=0.5)
    parser.add_argument('--holdout-age',
        type=int,
        default=40)
    parser.add_argument('--eps',
        type=float,
        help="outlier detector eps",
        default=0.1)
    parser.add_argument('--train-data-file',
        type=str,
        default="_output/train_data.pkl")
    parser.add_argument('--test-data-file',
        type=str,
        default="_output/test_data.pkl")
    parser.add_argument('--ulm-files',
        type=str,
        default="")
    parser.add_argument('--plain-files',
        type=str,
        default="")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument('--out-results',
        type=str,
        default="_output/out.pkl")
    parser.set_defaults()
    args = parser.parse_args()
    args.ulm_files = process_params(args.ulm_files, str)
    args.plain_files = process_params(args.plain_files, str)
    return args

def load_all_models(args, train_data):
    ulm_models = [load_model(ulm_file) for ulm_file in args.ulm_files]
    raw_plain_models = [load_model(plain_file) for plain_file in args.plain_files]
    plain_models = [
            DecisionPredictionModel(m, args.cost_decline)
            for m in raw_plain_models]
    od_models = [
            EntropyOutlierPredictionModel(m, args.cost_decline, eps=args.eps)
            for m in raw_plain_models]
    for m in od_models:
        m.fit_decision_model(train_data.x)

    return {
            "ulm": ulm_models,
            "plain": plain_models,
            "od": od_models}

def get_data(model_list, young_age_test_data, old_age_test_data, label):
    print(label)
    res_data = []
    for m_idx, m in enumerate(model_list):
        m_label = "%s_%d" % (label, m_idx)
        if m is None:
            logging.info("%s model_%d NOT AVAIL", label, m_idx)
            continue

        score = m.score(old_age_test_data.x, old_age_test_data.y)
        accept_prob = m.get_accept_prob(old_age_test_data.x)
        res_data += extract_row(score, "score old", m_label)
        res_data += extract_row(np.mean(accept_prob), "accept old", m_label)

        score = m.score(young_age_test_data.x, young_age_test_data.y)
        accept_prob = m.get_accept_prob(young_age_test_data.x)
        res_data += extract_row(score, "score young", m_label)
        res_data += extract_row(np.mean(accept_prob), "accept young", m_label)
    return res_data

def plot_results(model_dict, young_age_test_data, old_age_test_data):
    all_data = []
    for label, models in model_dict.items():
        new_data = get_data(models, young_age_test_data, old_age_test_data, label)
        all_data += new_data
    data_df = pd.DataFrame(all_data)
    logging.info(data_df.to_latex())
    print(data_df)
    return data_df

def create_holdout_datasets(test_data, args):
    scoring_test_data = Dataset(
            test_data.x[:,1:],
            test_data.y)
    ages = test_data.x[:,0]
    unseen_age_test_data = scoring_test_data.subset(ages > args.holdout_age)
    seen_ages_test_data = scoring_test_data.subset(ages <= args.holdout_age)
    return seen_ages_test_data, unseen_age_test_data

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_data = pickle_from_file(args.test_data_file)
    train_data_dict = pickle_from_file(args.train_data_file)
    train_data = train_data_dict["train"]
    support_sim_settings = train_data_dict["support_sim_settings"]
    check_supp = np.array(support_sim_settings.check_obs_x(test_data.x[:,1:]))
    print(check_supp.shape)
    test_data = test_data.subset(
            np.array(support_sim_settings.check_obs_x(test_data.x[:,1:])))

    model_dict = load_all_models(args, train_data)
    seen_ages_test_data, unseen_age_test_data = create_holdout_datasets(test_data, args)
    result_df = plot_results(
            model_dict,
            seen_ages_test_data,
            unseen_age_test_data)
    pickle_to_file(result_df, args.out_results)

if __name__ == "__main__":
    main(sys.argv[1:])
