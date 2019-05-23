"""
Plot mimic-related stuff
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from decision_interval_aggregator import DecisionIntervalAggregator
from decision_interval_recalibrator import DecisionIntervalRecalibrator
from dataset import Dataset
from common import pickle_to_file, pickle_from_file, process_params, get_normal_ci, load_model, get_interval_width
from plot_common import *

def plot_accept_prob_vs_missing(fitted_models, dataset, args):
    num_missing_indicators = int((dataset.x.shape[1] - 3)/2)
    print('num missing indicators', num_missing_indicators)
    missingness = np.mean(dataset.x[:, -num_missing_indicators:], axis=1)

    all_dfs = []
    for model_idx, fitted_model in enumerate(fitted_models):
        accept_probs = fitted_model.get_accept_prob(dataset.x)
        df = pd.DataFrame.from_dict({
            "missingness": missingness,
            "accept_prob": accept_probs.flatten(),
            "model_idx": model_idx})
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)

    plt.clf()
    sns.lineplot(
            x="missingness",
            y="accept_prob",
            hue="model_idx",
            data=all_dfs)
    plt.savefig(args.out_missing_plot)

def plot_accept_prob_vs_age(fitted_models, dataset, args):
    # Age is the first column!
    age = dataset.x[:,0].astype(int)
    all_dfs = []
    for model_idx, fitted_model in enumerate(fitted_models):
        accept_probs = fitted_model.get_accept_prob(dataset.x)
        df = pd.DataFrame.from_dict({
            "age": age,
            "accept_prob": accept_probs.flatten(),
            "Fold": model_idx})
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    print("num young peeps", np.sum(age < 18))

    plt.clf()
    plt.figure(figsize=(4,4))
    sns.set(style="white")
    sns.despine()
    sns.lineplot(
            x="age",
            y="accept_prob",
            hue="Fold",
            data=all_dfs)
    plt.xlabel("Age")
    plt.ylabel("Acceptance probability")
    plt.tight_layout()
    plt.savefig(args.out_age_plot)

def plot_prediction_vs_missingness(fitted_models, dataset, args):
    num_missing_indicators = int((dataset.x.shape[1] - 3)/2)
    print('num missing indicators', num_missing_indicators)
    missingness = np.mean(dataset.x[:, -num_missing_indicators:], axis=1)

    all_dfs = []
    for model_idx, fitted_model in enumerate(fitted_models):
        if not fitted_model.has_density:
            alpha_PIs = fitted_model.get_prediction_interval(dataset.x, args.alpha)
            prediction = np.mean(alpha_PIs, axis=1)
        elif fitted_model.density_parametric_form == "bernoulli":
            prediction = fitted_model.get_prediction_probs(dataset.x)[:,0]
        else:
            raise ValueError("not an option")
        accept_probs = fitted_model.get_accept_prob(dataset.x).flatten()
        accepted_mask = accept_probs > 0.5
        df = pd.DataFrame.from_dict({
            "missingness": missingness[accepted_mask],
            "prediction": prediction[accepted_mask],
            "model_idx": model_idx})
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)

    plt.clf()
    ax = sns.lineplot(
            x="missingness",
            y="prediction",
            hue="model_idx",
            data=all_dfs,
            ci="sd")
    plt.savefig(args.out_missing_pred_plot)

def plot_prediction_vs_age(
        fitted_models,
        dataset,
        args,
        y_label,
        pred_func=None,
        accept_prob_thres=0.5,
        max_predict_plot=False):
    """
    @param pred_func: whether to transform the prediction before plotting
    @param max_predict_plot: whether to only plot things within one std dev of the median
    """
    # Age is the first column!
    age = dataset.x[:,0].astype(int)
    all_dfs = []
    for model_idx, fitted_model in enumerate(fitted_models):
        if not fitted_model.has_density:
            alpha_PIs = fitted_model.get_prediction_interval(dataset.x, args.alpha)
            prediction = np.mean(alpha_PIs, axis=1)
        elif fitted_model.density_parametric_form == "bernoulli":
            prediction = fitted_model.get_prediction_probs(dataset.x)[:,0]
        else:
            raise ValueError("not an option")

        accept_probs = fitted_model.get_accept_prob(dataset.x).flatten()
        if max_predict_plot:
            accepted_mask = (accept_probs > accept_prob_thres) * (prediction < np.median(prediction) + np.sqrt(np.var(prediction)))
        else:
            accepted_mask = (accept_probs > accept_prob_thres)
        df = pd.DataFrame.from_dict({
            "Age": age[accepted_mask],
            "prediction": prediction[accepted_mask],
            "Fold": model_idx})
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    if pred_func is not None:
        all_dfs.prediction = pred_func(all_dfs.prediction)

    plt.clf()
    plt.figure(figsize=(4,4))
    sns.set(style="white")
    sns.despine()
    sns.lineplot(
            x="Age",
            y="prediction",
            hue="Fold",
            data=all_dfs,
            ci="sd")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(args.out_age_pred_plot)

#def plot_local_coverages(aggregator, dataset, args, fontsize=14):
#    """
#    Plot local coverages....
#    """
#    all_local_coverages = []
#    for model_idx, fitted_model in enumerate(aggregator.fitted_models):
#        local_coverages = assess_local_coverage(
#                fitted_model,
#                dataset,
#                alpha=args.alpha,
#                num_rand=args.num_rand,
#                k=args.num_nearest_neighbors)
#        print("local coverages assessed at %d points" % len(local_coverages))
#        all_local_coverages.append(pd.DataFrame.from_dict({
#            "model_idx": model_idx,
#            "local_coverage": local_coverages}))
#    local_agg_coverages = assess_local_agg_coverage(
#            aggregator,
#            dataset,
#            alpha=args.alpha,
#            num_rand=args.num_rand,
#            k=args.num_nearest_neighbors)
#    print(local_agg_coverages)
#    all_local_coverages.append(pd.DataFrame.from_dict({
#        "model_idx": "Agg",
#        "local_coverage": local_coverages}))
#
#    plt.clf()
#    sns.violinplot(
#            x="model_idx",
#            y="local_coverage",
#            data=pd.concat(all_local_coverages),
#            cut=0)
#    plt.xlabel("Fitted Model", fontsize=fontsize)
#    plt.ylabel("Local coverage", fontsize=fontsize)
#    plt.savefig(args.out_local_coverage_plot)
