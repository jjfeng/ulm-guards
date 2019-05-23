"""
Fits the decision function and density function simultaneously
"""
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np

from tune_hyperparams import do_cross_validation
from decision_density_nn import SimultaneousDensityDecisionNNs
from ensemble_decision_density_nn import EnsembleSimultaneousDensityDecisionNNs
from support_sim_settings import SupportSimSettingsEmpirical
from common import pickle_to_file, pickle_from_file, process_params, make_scratch_dir


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
    parser.add_argument('--num-ensemble',
        type=int,
        default=1)
    parser.add_argument('--num-inits',
        type=int,
        help="num random initializations",
        default=1)
    parser.add_argument('--support-sim-num',
        type=int,
        help="number of points to sample from the support (uniformly)",
        default=100)
    parser.add_argument('--data-file',
        type=str,
        help="input data file name, needs to be a pickle file containing a Dataset obj (see data_generator.py)",
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        help="data split file",
        default="_output/data_split.pkl")
    parser.add_argument('--log-barrier-params',
        type=str,
        help="""
        The weight param for the log barrier penalty
        comma separated list (for tuning over)
        """,
        default="0.01")
    parser.add_argument('--do-no-harm-params',
        type=str,
        help="""
        The weight param for the do-no-harm penalty
        comma separated list (for tuning over)
        """,
        default="0.1")
    parser.add_argument('--cost-decline',
        type=float,
        help=""""
        The cost of declining to make a prediction.
        If is zero, we assume that we actually want an accept ALL model (not accept none).
        """,
        default=0.1)
    parser.add_argument('--density-weight-params',
        type=str,
        help="""
        Ridge penalty of the coefficients in the neural network parameterizing the conditioanl density
        comma separated list of weight params (for tuning over)
        """,
        default="0.1")
    parser.add_argument('--decision-weight-params',
        type=str,
        help="""
        Ridge penalty of the coefficients in the neural network parameterizing the decision function
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
    parser.add_argument('--density-layer-sizes',
        type=str,
        help="""
        NN structure for density function
        comma-colon separated lists of +-separated lists of layer sizes (for tuning over)
        does not include the output layer
        Example: 6+2+1,6+2+2+1
        """,
        default="4+10")
    parser.add_argument('--decision-layer-sizes',
        type=str,
        help="""
        NN structure for density function
        comma-colon separated lists of +-separated lists of layer sizes (for tuning over)
        does not include the output layer
        Example: 6+2+1,6+2+2+1
        """,
        default=None)
    parser.add_argument('--dropout-rate',
        type=str,
        help="""
        dropout rate (0 means no dropout)
        comma separated list (for tuning over)
        """,
        default="0")
    parser.add_argument('--density-parametric-form',
        type=str,
        default="gaussian",
        help="The parametric form we are going to use for Y|X",
        choices=["gaussian", "bernoulli", "multinomial"])
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
    parser.add_argument('--use-train-data-support',
        action="store_true")
    parser.add_argument('--empirical-support-file',
        type=str,
        default="",
        help="support empirical distribution")
    parser.add_argument('--support-empirical-scale',
        type=float,
        default=1,
        help="How much to scale the gaussian perturbation around the empirical support distirbution")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--do-distributed',
        action="store_true")
    parser.add_argument('--scratch-dir',
        type=str,
        default="scratch")
    parser.set_defaults()
    args = parser.parse_args()

    # Parse parameters to CV over
    args.do_no_harm_params = process_params(args.do_no_harm_params, float)
    args.log_barrier_params = process_params(args.log_barrier_params, float)
    args.density_weight_params = process_params(args.density_weight_params, float)
    args.decision_weight_params = process_params(args.decision_weight_params, float)
    args.dropout_rate = process_params(args.dropout_rate, float)
    args.density_layer_sizes = [process_params(substr, str, "+") for substr in args.density_layer_sizes.split(",")]
    if args.decision_layer_sizes is not None:
        args.decision_layer_sizes = [process_params(substr, str, "+") for substr in args.decision_layer_sizes.split(",")]
    else:
        args.decision_layer_sizes = [None]
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)
    nn_class = EnsembleSimultaneousDensityDecisionNNs

    scratch_dir = make_scratch_dir(args)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Read data
    data_dict = pickle_from_file(args.data_file)
    #assert data_dict["support_sim_settings"].check_dataset(data_dict["train"])
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    print(split_dict["train_idxs"])
    print(data_dict["train"].x.shape)
    print(data_dict["train"].y.shape)
    print(split_dict["train_idxs"].shape)
    train_split_dataset = data_dict["train"].subset(split_dict["train_idxs"])

    print(train_split_dataset.y.shape)
    if args.density_parametric_form == "multinomial":
        print("num classes", train_split_dataset.num_classes)
        density_parametric_form = "multinomial%d" % train_split_dataset.num_classes
    else:
        density_parametric_form = args.density_parametric_form

    if args.use_train_data_support:
        support_data = train_split_dataset
        old_support_settings = data_dict["support_sim_settings"]
        data_dict["support_sim_settings"] = SupportSimSettingsEmpirical(
                support_data.x,
                scale=args.support_empirical_scale,
                min_x=old_support_settings.min_x,
                max_x=old_support_settings.max_x)
    elif args.empirical_support_file:
        empirical_support = pickle_from_file(args.empirical_support_file)
        old_support_settings = data_dict["support_sim_settings"]
        data_dict["support_sim_settings"] = SupportSimSettingsEmpirical(
                empirical_support,
                scale=args.support_empirical_scale,
                min_x=old_support_settings.min_x,
                max_x=old_support_settings.max_x)

    # Setup the parameters we will tune over
    param_grid = [{
        'density_layer_sizes': args.density_layer_sizes,
        'decision_layer_sizes': args.decision_layer_sizes,
        'dropout_rate': args.dropout_rate,
        'density_parametric_form': [density_parametric_form],
        'density_weight_param': args.density_weight_params,
        'decision_weight_param': args.decision_weight_params,
        'weight_penalty_type': [args.weight_penalty_type],
        'cost_decline': [args.cost_decline],
        'do_no_harm_param': args.do_no_harm_params,
        'log_barrier_param': args.log_barrier_params,
        'max_iters': [args.max_iters],
        'num_inits': [args.num_inits],
        'num_ensemble': [args.num_ensemble],
        'do_distributed': [args.do_distributed],
        'scratch_dir': [scratch_dir],
        'act_func': [args.act_func],
        'learning_rate': [args.learning_rate],
        'support_sim_settings': [data_dict["support_sim_settings"]],
        'support_sim_num': [args.support_sim_num],
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
        "fitted_params": [m.model_params for m in fitted_model.nns],
        "hyperparams": best_hyperparams,
        "cv_results": cv_results,
    }, args.fitted_file)

    # DOUBLE CHECKING THINGS WORK
    #pickle_from_file(args.fitted_file)
    fitted_model.get_accept_prob(train_split_dataset.x[:10,:])
    fitted_model.get_prediction_interval(train_split_dataset.x[:10,:])

if __name__ == "__main__":
    main(sys.argv[1:])
