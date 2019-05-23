from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import scipy.stats
from sklearn.metrics import roc_auc_score

from dataset import Dataset
from data_generator import DataGenerator
from decision_prediction_base_model import DecisionPredictionBaseModel
from decision_interval_aggregator import DecisionIntervalAggregator
from common import is_within_interval, get_interval_width, distance_from_interval

def assess_local_agg_coverage_true(
        decision_interval_agg: DecisionIntervalAggregator,
        dataset: Dataset,
        data_generator: DataGenerator,
        accept_thres: float = 0.7):
    all_accept_PIs = decision_interval_agg.aggregate_prediction_intervals(dataset.x)
    true_mus = data_generator.mu_func(dataset.x)
    true_sigmas = data_generator.sigma_func(dataset.x)
    num_models = all_accept_PIs.shape[0]
    agg_model_coverage = []
    agg_model_accept = np.mean(all_accept_PIs[:,:,0], axis=0)
    individual_coverages = []
    for i in range(num_models):
        indiv_accept_probs = all_accept_PIs[i,:,0]
        mask = indiv_accept_probs > accept_thres
        indiv_PIs = all_accept_PIs[i,:,1:]
        lower_cdf = scipy.stats.norm.cdf(indiv_PIs[:,0], loc=true_mus, scale=true_sigmas)
        upper_cdf = scipy.stats.norm.cdf(indiv_PIs[:,1], loc=true_mus, scale=true_sigmas)
        actual_indiv_coverage = upper_cdf - lower_cdf
        accepted_indiv_cov = actual_indiv_coverage[mask]
        individual_coverages.append(accepted_indiv_cov)
        agg_model_coverage.append(actual_indiv_coverage)

    agg_model_coverage = np.mean(agg_model_coverage, axis=0)
    agg_mask = agg_model_accept > accept_thres
    accepted_agg_cov = agg_model_coverage[agg_mask]

    res = {
            "local_coverage_var": {
                "agg": [np.var(accepted_agg_cov)],
                "individual": [np.var(indiv_local_cov) for indiv_local_cov in individual_coverages],
            },
            "local_coverage_iqr": {
                "agg": [scipy.stats.iqr(accepted_agg_cov)],
                "individual": [scipy.stats.iqr(indiv_local_cov) for indiv_local_cov in individual_coverages],
            }
        }
    print(res)
    return res

def assess_local_agg_coverage(
        decision_interval_agg: DecisionIntervalAggregator,
        dataset: Dataset,
        num_rand: int = 20,
        k: int =10,
        accept_thres: float = 0.5):
    """
    @return array of the local coverage among the `k` nearest neighbors of `num_rand` points selected
                from the dataset
    """
    # Use KDTree to find the k closest points (currently using euclidean norm)
    tree = KDTree(dataset.x)
    rand_idxs = np.random.permutation(np.arange(dataset.num_obs))[:num_rand]
    nearest_neighbors_idxs = tree.query(dataset.x[rand_idxs], k=k, return_distance=False)

    # Now assess local coverage for each local sampling
    local_indiv_coverages = []
    local_agg_coverages = []
    for idx in range(num_rand):
        local_idxs = nearest_neighbors_idxs[idx,:]
        local_data = dataset.subset(local_idxs)
        eval_local_coverage = decision_interval_agg.eval_cov_given_accept(local_data)

        accept_indivs = eval_local_coverage["accept"]["individual"]
        cov_given_accept_indivs = eval_local_coverage["cov_given_accept"]["individual"]
        for model_idx, (accept_indiv, cov_given_accept_indiv) in enumerate(zip(accept_indivs, cov_given_accept_indivs)):
            if accept_indiv > accept_thres:
                local_indiv_coverages.append({
                    "model_idx": model_idx,
                    "local_coverage": cov_given_accept_indiv})

        mean_accept_prob = eval_local_coverage["accept"]["mean"]
        print(idx, mean_accept_prob)
        if mean_accept_prob > accept_thres:
            local_agg_coverages.append({
                "model_idx": "Agg",
                "local_coverage": eval_local_coverage["cov_given_accept"]["mean"]})
    return pd.DataFrame(local_agg_coverages + local_indiv_coverages)

def get_avg_accept_prob(
        fitted_models: List[DecisionPredictionBaseModel],
        dataset: Dataset):
    accept_prob = np.concatenate([
        fitted_model.get_accept_prob(dataset.x).reshape((-1,1))
        for fitted_model in fitted_models],axis=1)
    return np.mean(accept_prob, axis=1)

def get_interval_loss_given_accept(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset,
        alpha: float):
    alpha_PIs = model_to_eval.get_prediction_interval(dataset.x, alpha)
    interval_dist = distance_from_interval(alpha_PIs, dataset.y).flatten()
    interval_width = get_interval_width(alpha_PIs).flatten()
    interval_loss = alpha * interval_width/2 + interval_dist

    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    return np.sum(interval_loss * accept_prob)/np.sum(accept_prob)

def get_accept_prob(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset):
    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    return np.mean(accept_prob)

def get_max_log_lik_given_accept(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset,
        eps: float = 1e-30):
    pdf_y_given_x = model_to_eval.get_density(dataset.x, dataset.y).flatten() + eps
    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    #print(np.log(pdf_y_given_x)[accept_prob > 0.8])
    return np.min(np.log(pdf_y_given_x) * accept_prob - 4 * (1 - accept_prob))

def get_log_lik_given_accept(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset,
        eps: float = 1e-30):
    pdf_y_given_x = model_to_eval.get_density(dataset.x, dataset.y).flatten() + eps
    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    return np.sum(np.log(pdf_y_given_x) * accept_prob)/np.sum(accept_prob)

def get_pi_width_given_accept(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset,
        alpha: float):
    alpha_PIs = model_to_eval.get_prediction_interval(dataset.x, alpha)
    pi_width = get_interval_width(alpha_PIs)
    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    return np.sum(pi_width * accept_prob)/np.sum(accept_prob)

def get_coverage_given_accept(
        model_to_eval: DecisionPredictionBaseModel,
        dataset: Dataset,
        alpha: float):
    alpha_PIs = model_to_eval.get_prediction_interval(dataset.x, alpha)
    in_interval = is_within_interval(alpha_PIs, dataset.y)
    accept_prob = model_to_eval.get_accept_prob(dataset.x).flatten()
    return np.sum(in_interval * accept_prob)/np.sum(accept_prob)

def get_auc(model, should_accept_x, should_reject_x):
    entropy1 = model.get_univar_mapping(should_accept_x).reshape((-1,1))
    entropy0 = model.get_univar_mapping(should_reject_x).reshape((-1,1))
    decision_vals = np.concatenate([entropy1, entropy0])
    true_accept_labels = np.zeros((should_accept_x.shape[0] + should_reject_x.shape[0], 1))
    true_accept_labels[should_accept_x.shape[0]:] = 1
    decision_vals[np.isinf(decision_vals)] = np.min(decision_vals[np.isfinite(decision_vals)])
    decision_vals[np.isnan(decision_vals)] = 0
    #print('decision vals', decision_vals)
    auc = roc_auc_score(true_accept_labels, decision_vals)
    return auc

def extract_row(val: float, key: str, type_str: str):
    try:
        return [{
            "key": key,
            "value": float(val),
            "type": type_str}]
    except TypeError:
        print(val, key, type_str)
        print("failed casting to float")
