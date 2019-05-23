"""
Create data
"""
import sys
import argparse
import logging
import numpy as np

from data_generator import DataGenerator
from common import pickle_to_file


def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--num-p',
        type=int,
        help="Dimension of X",
        default=4)
    parser.add_argument('--std-dev-x',
        type=int,
        help="std dev of X (assuming normal distribution)",
        default=1)
    parser.add_argument('--max-x',
        type=int,
        help="max X for support (min X will be the negative)",
        default=10)
    parser.add_argument('--num-train',
        type=int,
        help="Num training samples",
        default=2500)
    parser.add_argument('--sim-func-form',
        type=str,
        help="type of parametric form of Y|X",
        default="gaussian")
    parser.add_argument('--num-classes',
        type=int,
        help="number of classes in multinomial (ignored for other sim funcs)",
        default=1)
    parser.add_argument('--sim-func',
        type=str,
        help="what to simulate from",
        default="simple")
    parser.add_argument('--sim-noise-sd',
        type=float,
        help="how much to scale the noise",
        default=1)
    parser.add_argument('--out-data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    return args


def main(args=sys.argv[1:]):
    args = parse_args()
    print(args)

    np.random.seed(args.seed)

    data_gen = DataGenerator(
        sim_func_form=args.sim_func_form,
        sim_func_name=args.sim_func,
        num_p=args.num_p,
        num_classes=args.num_classes,
        noise_sd=args.sim_noise_sd,
        std_dev_x=args.std_dev_x,
        max_x=args.max_x,
    )
    train_data, support_sim_settings = data_gen.create_data(args.num_train)

    # Write data to file
    pickle_to_file({
        "train": train_data,
        "support_sim_settings": support_sim_settings,
        "data_gen": data_gen
    }, args.out_data_file)


if __name__ == "__main__":
    main(sys.argv[1:])
