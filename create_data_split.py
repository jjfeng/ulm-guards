"""
Split data
"""
import sys
import argparse
import logging
import numpy as np

from data_generator import Dataset
from common import pickle_to_file, pickle_from_file


def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--k-folds',
        type=int,
        help="number of folds",
        default=3)
    parser.add_argument('--fold-idx',
        type=int,
        help='number of fold in the k-folds to hold out for recalibration',
        default=1)
    parser.add_argument('--recalibrate-num',
        type=int,
        help="number to put in recalibration (choose either proportion or num)",
        default=None)
    parser.add_argument('--in-data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/data_split.pkl")
    parser.set_defaults()
    args = parser.parse_args()
    if args.recalibrate_num is not None:
        assert args.recalibrate_num > 0
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    # Read data
    data_dict = pickle_from_file(args.in_data_file)
    full_data = data_dict["train"]
    unique_groups = np.unique(full_data.group_id)
    shuffled_order = np.random.permutation(unique_groups)

    if args.recalibrate_num is not None:
        num_recalibrate = args.recalibrate_num
        train_groups = shuffled_order[:-num_recalibrate]
        recalibrate_groups = shuffled_order[-num_recalibrate:]
    else:
        fold_size = int(unique_groups.size/args.k_folds) + 1
        start_idx = args.fold_idx * fold_size
        end_idx = min((args.fold_idx + 1) * fold_size, unique_groups.size)
        print("number in recalibrated groups", end_idx - start_idx)
        train_groups = np.concatenate([
                shuffled_order[:start_idx],
                shuffled_order[end_idx:]])
        recalibrate_groups = shuffled_order[start_idx: end_idx]

    train_idxs = np.isin(full_data.group_id, train_groups).flatten()
    assert train_idxs.size > 1

    # For recalibartion, we only grab a random obs per group
    recalibrate_idxs = []
    for recalib_group_id in recalibrate_groups:
        matching_obs_idxs = np.where(full_data.group_id == recalib_group_id)[0]
        random_matching_obs_idx = np.random.choice(matching_obs_idxs)
        recalibrate_idxs.append(random_matching_obs_idx)
    recalibrate_idxs = np.array(recalibrate_idxs)

    assert recalibrate_idxs.size > 1
    # Double check we grabbed a single random obs per group
    assert np.unique(full_data.group_id[recalibrate_idxs]).size == recalibrate_idxs.size

    # Write data to file
    print("num train", train_idxs.size)
    pickle_to_file({
        "train_idxs": train_idxs,
        "recalibrate_idxs": recalibrate_idxs,
        "support_sim_settings": data_dict["support_sim_settings"],
    }, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
