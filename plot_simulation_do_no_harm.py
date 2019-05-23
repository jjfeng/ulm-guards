"""
visualize how validation loss and acceptance probability
change with different lambda values (lambda = do no harm regularization)
"""
import sys
import argparse
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from common import load_model, pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--num-test',
        type=int,
        default=100)
    parser.add_argument('--data-split-file',
        type=str,
        default="_output/data_split.pkl")
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl",
        help="comma separated")
    parser.add_argument('--plot-val-loss-file',
        type=str,
        default="_output/plot_lambda.png")
    parser.add_argument('--plot-support-val-loss-file',
        type=str,
        default="_output/plot_support_unif_lambda.png")
    parser.add_argument('--plot-accept-region-file',
        type=str,
        default="_output/plot_accept_region_lambda.png")
    parser.add_argument('--plot-accept-file',
        type=str,
        default="_output/plot_accept_lambda.png")
    parser.add_argument('--plot-support-accept-file',
        type=str,
        default="_output/plot_accept_unif_lambda.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    return args

def plot_accepted_rejected_region(data_dict, fitted_models, args, mesh_size=0.05):
    """
    Plot acceptance probability. Last row plot pdf of X
    """
    COLORS = ['orange', 'red']
    LINESTYLES = ['dashed', 'dotted']

    num_models = len(fitted_models)
    # Look at the region we accepted
    mesh_coords, (xx, yy) = data_dict["support_sim_settings"].generate_grid(mesh_size)
    x_pdf = data_dict["data_gen"].get_x_pdf(mesh_coords)
    all_accept_probs = []
    for fitted_model in fitted_models:
        x_accept_probs = fitted_model.get_accept_prob(mesh_coords)
        all_accept_probs.append(x_accept_probs)

    fig, ax = plt.subplots(nrows=1, figsize=(4,4))
    for idx, x_accept_probs in enumerate(all_accept_probs):
        print("MIN ACC", np.min(x_accept_probs))
        cs = ax.contour(
                xx,
                yy,
                x_accept_probs.reshape(xx.shape),
                levels=[0.99],
                colors=COLORS[idx],
                linestyles=LINESTYLES[idx],
                linewidths=4)
    cs = ax.contourf(xx, yy, x_pdf.reshape(xx.shape), cmap='gray')
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(args.plot_accept_region_file)


def plot_validation_losses(fitted_models, data, args):
    validation_losses = []
    lambdas = []
    for fitted_model in fitted_models:
        validation_loss = -fitted_model.score(data.x, data.y)
        validation_losses.append(validation_loss)
        lambdas.append(fitted_model.do_no_harm_param)

    print("loss argmin", np.argmin(validation_losses))
    print("val loss", validation_losses)
    plt.clf()
    sns.regplot(
            lambdas,
            validation_losses,
            fit_reg=False)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.savefig(args.plot_val_loss_file)

def plot_validation_losses_support_unif(
        fitted_models,
        support_sim_settings,
        data_gen,
        args):
    sim_support_x = support_sim_settings.support_unif_rvs(args.num_test)
    support_unif_data = data_gen.create_data_given_x(sim_support_x)

    validation_losses = []
    lambdas = []
    for fitted_model in fitted_models:
        validation_loss = -fitted_model.score(support_unif_data.x, support_unif_data.y)
        validation_losses.append(validation_loss)
        lambdas.append(fitted_model.do_no_harm_param)

    print("unif loss argmin", np.argmin(validation_losses))
    print("unif val loss", validation_losses)
    plt.clf()
    sns.regplot(
            lambdas,
            validation_losses,
            fit_reg=False)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.savefig(args.plot_support_val_loss_file)

def plot_accept_probs(
        fitted_models,
        data,
        args):
    all_accept_probs = []
    lambdas = []
    for fitted_model in fitted_models:
        accept_probs = fitted_model.get_accept_prob(data.x)
        all_accept_probs.append(np.mean(accept_probs))
        lambdas.append(fitted_model.do_no_harm_param)

    print("accept", all_accept_probs)
    plt.clf()
    sns.regplot(
            lambdas,
            all_accept_probs,
            fit_reg=False)
    #plt.xscale("log")
    plt.savefig(args.plot_accept_file)

def plot_accept_probs_support_unif(
        fitted_models,
        support_sim_settings,
        data_gen,
        args):
    sim_support_x = support_sim_settings.support_unif_rvs(args.num_test)

    all_accept_probs = []
    lambdas = []
    for fitted_model in fitted_models:
        accept_probs = fitted_model.get_accept_prob(sim_support_x)
        all_accept_probs.append(np.mean(accept_probs))
        lambdas.append(fitted_model.do_no_harm_param)
    print("lambdas", lambdas)
    print("uniform accept", all_accept_probs)
    plt.clf()
    sns.regplot(
            lambdas,
            all_accept_probs,
            fit_reg=False)
    #plt.xscale("log")
    plt.savefig(args.plot_support_accept_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    # Read all data
    orig_data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    recalib_data = orig_data_dict["train"].subset(split_dict["recalibrate_idxs"])
    args.num_p = recalib_data.x.shape[1]

    # Load models
    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]

    # Do all the plotting
    if args.num_p == 2:
        plot_accepted_rejected_region(orig_data_dict, fitted_models, args)
    #plot_validation_losses(fitted_models, recalib_data, args)
    #plot_validation_losses_support_unif(
    #        fitted_models,
    #        orig_data_dict["support_sim_settings"],
    #        orig_data_dict["data_gen"],
    #        args)
    #plot_accept_probs(fitted_models, recalib_data, args)
    #plot_accept_probs_support_unif(
    #        fitted_models,
    #        orig_data_dict["support_sim_settings"],
    #        orig_data_dict["data_gen"],
    #        args)

if __name__ == "__main__":
    main(sys.argv[1:])
