"""
visualize mnist
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List

from dataset import Dataset
from decision_prediction_combo import *
from common import *
from plot_common import *

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',
        type=int,
        default=0,
        help="random seed")
    parser.add_argument('--plain-eps',
        type=float,
        default=0.1,
        help="eps for outlier detector")
    parser.add_argument('--ensemble-eps',
        type=float,
        default=0.1,
        help="eps for outlier detector")
    parser.add_argument('--cost-decline',
        type=float,
        default=1)
    parser.add_argument('--ulm-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--ensemble-ulm-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--plain-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--dropout-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--ensemble-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--train-data-file',
        type=str,
        default="../data/mnist/mnist_train_pca.pkl")
    parser.add_argument('--test-data-file',
        type=str,
        default="../data/mnist/mnist_test_pca.pkl")
    parser.add_argument('--weird-data-file',
        type=str,
        default="../data/mnist/weird_mnist_test_pca.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/plot_mnist_log.txt")
    parser.add_argument('--out-results-file',
        type=str,
        default="_output/results.pkl")
    parser.set_defaults()
    args = parser.parse_args()
    args.ulm_files = process_params(args.ulm_files, str)
    args.ensemble_ulm_files = process_params(args.ensemble_ulm_files, str)
    args.ensemble_files = process_params(args.ensemble_files, str)
    args.dropout_files = process_params(args.dropout_files, str)
    args.plain_files = process_params(args.plain_files, str)
    return args

def get_data(models, test_data, num_train_classes, weird_x, random_x, label):
    print(label)
    unseen_idxs = np.where(test_data.y > 0)[1] >= num_train_classes
    seen_idxs = np.logical_not(unseen_idxs)
    res_data = []
    # Collect results for the simultaneous decision density NNs
    for idx, model in enumerate(models):
        accept_probs = model.get_accept_prob(test_data.x).reshape((-1,1))
        accept_seen = np.mean(accept_probs[seen_idxs])
        accept_unseen = np.mean(accept_probs[unseen_idxs])

        seen_test_data = test_data.subset(seen_idxs)
        seen_test_data.y = seen_test_data.y[:,:num_train_classes]
        seen_test_data.num_classes = num_train_classes

        #auc = get_auc(model, test_data.x[seen_idxs], test_data.x[unseen_idxs])

        score_seen = model.score(test_data.x[seen_idxs], test_data.y[seen_idxs, :num_train_classes])

        accept_weird = model.get_accept_prob(weird_x).mean()
        #weird_auc = get_auc(model, test_data.x[seen_idxs], weird_x)

        accept_random = model.get_accept_prob(random_x).mean()
        #random_auc = get_auc(model, test_data.x[seen_idxs], random_x)

        res_data += extract_row(accept_seen, "accept_seen", label)
        res_data += extract_row(accept_unseen, "accept_unseen", label)
        res_data += extract_row(score_seen, "score_seen", label)
        #res_data += extract_row(auc, "auc", label)
        res_data += extract_row(accept_weird, "accept_weird", label)
        #res_data += extract_row(weird_auc, "weird_auc", label)
        res_data += extract_row(accept_random, "accept_random", label)
        #res_data += extract_row(random_auc, "random_auc", label)
    return res_data

def plot_results(
        result_dict: Dict[str, List],
        test_data: Dataset,
        num_train_classes: int,
        weird_x: ndarray,
        random_x: ndarray):
    all_data = []
    for key, val in result_dict.items():
        new_data = get_data(val, test_data, num_train_classes, weird_x, random_x, key)
        all_data += new_data
    data_df = pd.DataFrame(all_data)
    logging.info(data_df.to_latex())
    print(data_df)
    return data_df

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_dataset = pickle_from_file(args.test_data_file)
    train_data_dict = pickle_from_file(args.train_data_file)
    weird_x = pickle_from_file(args.weird_data_file)
    train_dataset = train_data_dict["train"]
    random_x = train_data_dict["support_sim_settings"].support_unif_rvs(100)

    nns_dict = {
        "ulm": args.ulm_files,
        "ensemble_ulm": args.ensemble_ulm_files,
        #"dropout": args.dropout_files,
        "plain": args.plain_files,
        "ensemble": args.ensemble_files,
    }
    for key, val in nns_dict.items():
        print(key)
        nns_dict[key] = [load_model(file_name) for file_name in val]

    decision_density_dict = {}
    for key, val in nns_dict.items():
        if "ulm" in key:
            decision_density_dict[key] = val
        else:
            decision_density_dict["%s+cutoff" % key] = [
                DecisionPredictionModel(nns, args.cost_decline)
                for nns in val]
    #decision_density_dict["ensemble+all"] = [
    #        AcceptAllPredictionModel(nns) for nns in nns_dict["ensemble"]]
    model_types = {
            "ensemble": args.ensemble_eps,
            "plain": args.plain_eps}
    for model_type, eps in model_types.items():
        combo_key = "%s+cutoff+OD%s" % (model_type, str(eps))
        decision_density_dict[combo_key] = [
                EntropyOutlierPredictionModel(nns, args.cost_decline, eps=eps)
                for nns in nns_dict[model_type]]
        for m in decision_density_dict[combo_key]:
            m.fit_decision_model(train_dataset.x)

    data_df = plot_results(
            decision_density_dict,
            test_dataset,
            train_dataset.num_classes,
            weird_x,
            random_x)
    pickle_to_file(data_df, args.out_results_file)

if __name__ == "__main__":
    main(sys.argv[1:])

