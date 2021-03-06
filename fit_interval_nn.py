"""
Fits the interval function, assumes all points are accepted
"""
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np

from tune_hyperparams import do_cross_validation
from ensemble_interval_nn import EnsembleIntervalNN
from common import pickle_to_file, pickle_from_file, process_params


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--max-iters',
        type=int,
        help="max training iters",
        default=2000)
    parser.add_argument('--num-inits',
        type=int,
        help="num random initializations",
        default=1)
    parser.add_argument('--data-file',
        type=str,
        help="input data file name, needs to be a pickle file containing a Dataset obj (see data_generator.py)",
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        help="data split file",
        default="_output/data_split.pkl")
    parser.add_argument('--interval-alpha',
        type=float,
        help="Train intervals to have coverage (1 - alpha)",
        default=0.1)
    parser.add_argument('--interval-weight-params',
        type=str,
        help="""
        Ridge penalty of the coefficients in the neural network parameterizing the conditioanl interval
        comma separated list of weight params (for tuning over)
        """,
        default="0.1")
    parser.add_argument('--weight-penalty-type',
        type=str,
        help="""
        Weight type lasso or ridge
        """,
        default="ridge",
        choices=["ridge", "group_lasso"])
    parser.add_argument('--interval-layer-sizes',
        type=str,
        help="""
        NN structure for interval function
        comma-colon separated lists of +-separated lists of layer sizes (for tuning over)
        does not include the output layer
        Example: 6+2+1,6+2+2+1
        """,
        default="4+10")
    parser.add_argument('--dropout-rate',
        type=str,
        help="""
        dropout rate (0 means no dropout)
        comma separated list (for tuning over)
        """,
        default="0")
    parser.add_argument('--act-func',
        type=str,
        help="activation function in all the NNs",
        default="tanh",
        choices=["tanh", "relu"])
    parser.add_argument('--learning-rate',
        type=float,
        help="learning rate for adam",
        default=0.002)
    parser.add_argument('--cv',
        type=int,
        help="num cross-validation folds",
        default=2)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/fitted.pkl")
    parser.set_defaults()
    args = parser.parse_args()

    # Parse parameters to CV over
    args.interval_weight_params = process_params(args.interval_weight_params, float)
    args.interval_layer_sizes = [process_params(substr, str, "+") for substr in args.interval_layer_sizes.split(",")]
    args.dropout_rate = process_params(args.dropout_rate, float)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    nn_class = EnsembleIntervalNN

    # Read data
    data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    train_split_dataset = data_dict["train"].subset(split_dict["train_idxs"])

    # Setup the parameters we will tune over
    param_grid = [{
        'interval_alpha': [args.interval_alpha],
        'interval_layer_sizes': args.interval_layer_sizes,
        'interval_weight_param': args.interval_weight_params,
        'dropout_rate': args.dropout_rate,
        'weight_penalty_type': [args.weight_penalty_type],
        'max_iters': [args.max_iters],
        'num_ensemble': [1],
        'num_inits': [args.num_inits],
        'act_func': [args.act_func],
        'learning_rate': [args.learning_rate],
    }]

    # Fit model
    fitted_model, best_hyperparams, cv_results = do_cross_validation(
        train_split_dataset,
        nn_class=nn_class,
        param_grid=param_grid,
        cv=args.cv)
    logging.info("Best hyperparams %s", best_hyperparams)

    # Save model
    pickle_to_file({
        "nn_class": nn_class,
        "fitted_params": [nn.model_params for nn in fitted_model.nns],
        "hyperparams": best_hyperparams,
        "cv_results": cv_results,
    }, args.fitted_file)

if __name__ == "__main__":
    main(sys.argv[1:])
