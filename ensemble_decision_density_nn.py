import time
import copy
import logging

import scipy.stats
import tensorflow as tf
import numpy as np
from numpy import ndarray

from nn_worker import NNWorker
from support_sim_settings import SupportSimSettings
from decision_density_nn import SimultaneousDensityDecisionNNs
from parallel_worker import *
from common import *

class EnsembleSimultaneousDensityDecisionNNs:
    """
    Fits a `decision_nn` that decides whether or not to accept
    Fits a `prediction_nn` for the conditional pdf p(y|x)
    """
    def __init__(self,
            density_layer_sizes=None,
            decision_layer_sizes=None,
            density_parametric_form="gaussian",
            dropout_rate=0,
            decision_weight_param=0.01,
            density_weight_param=0.01,
            weight_penalty_type="ridge",
            cost_decline=0.01,
            do_no_harm_param=0.01,
            log_barrier_param=0.01,
            num_inits=1,
            num_ensemble=1,
            max_iters=200,
            act_func: str="tanh",
            learning_rate: float=0.002,
            sgd_batch_size: int=4000,
            support_sim_settings: SupportSimSettings=None,
            support_sim_num: int=100,
            do_distributed=False,
            scratch_dir="/fh/fast/matsen_e/jfeng2/scratch"):
        """
        (other parameters defined in fit_simultaneous_decision_prediction_nn.py)
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param learning_rate: learning rate for adam optimization
        @param support_sim_settings: SupportSimSettings -- specifies how to simulate
                    points over the support of X
        @param support_sim_num: number of points to simulate at each step
                    of SGD to approximate the integral
        @param shared_layer: the layer that is shared by the decision and density nns
        """
        self.density_parametric_form = density_parametric_form
        self.density_layer_sizes = density_layer_sizes
        self.decision_layer_sizes = decision_layer_sizes
        self.dropout_rate = dropout_rate
        self.decision_weight_param = decision_weight_param
        self.density_weight_param = density_weight_param
        self.weight_penalty_type = weight_penalty_type
        self.cost_decline = cost_decline
        self.do_no_harm_param = do_no_harm_param
        self.log_barrier_param = log_barrier_param
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.num_ensemble = num_ensemble
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.sgd_batch_size = sgd_batch_size
        self.support_sim_settings = support_sim_settings
        self.support_sim_num = support_sim_num
        self.do_distributed = do_distributed
        self.scratch_dir = scratch_dir
        self._init_nn()

    def _init_nn(self, init_inner=False):
        self.nns = [SimultaneousDensityDecisionNNs(
            density_layer_sizes=self.density_layer_sizes,
            decision_layer_sizes=self.decision_layer_sizes,
            density_parametric_form=self.density_parametric_form,
            dropout_rate=self.dropout_rate,
            decision_weight_param=self.decision_weight_param,
            density_weight_param=self.density_weight_param,
            weight_penalty_type=self.weight_penalty_type,
            cost_decline=self.cost_decline,
            do_no_harm_param=self.do_no_harm_param,
            log_barrier_param=self.log_barrier_param,
            num_inits=self.num_inits,
            max_iters=self.max_iters,
            act_func=self.act_func,
            learning_rate=self.learning_rate,
            model_params=None,
            sgd_batch_size=self.sgd_batch_size,
            support_sim_settings=self.support_sim_settings,
            support_sim_num=self.support_sim_num)
            for _ in range(self.num_ensemble)]
        if init_inner:
            for nn in self.nns:
                nn._init_nn()

    def fit(self, X, y):
        worker_nns = [copy.deepcopy(nn) for nn in self.nns]
        for nn in self.nns:
            nn._init_nn()
        rand_seed = np.random.randint(1)
        worker_list = [
                NNWorker(rand_seed + idx, nn)
                for idx, nn in enumerate(worker_nns)]
        if self.do_distributed and len(worker_list) > 1:
            print("DISTRIBUTED")
            # Submit jobs to slurm
            batch_manager = BatchSubmissionManager(
                    worker_list=worker_list,
                    shared_obj=(X,y,None),
                    num_approx_batches=len(worker_list),
                    worker_folder=self.scratch_dir)
            results = batch_manager.run()
            print("RESULTS", results)
        else:
            # Run jobs locally
            print("LOCAL")
            results = [worker.run_worker((X, y, None)) for worker in worker_list]

        self.set_model_params(results)
        accept_probs = self.get_accept_prob(X)
        logging.info("MEAN ACCEPT %f", np.mean(accept_probs))

    def set_model_params(self, model_params_list):
        for nn, model_param in zip(self.nns, model_params_list):
            print("SETTING MODEL PARAMS")
            nn.set_model_params(model_param)

    def get_params(self, deep=True):
        return {
            "density_parametric_form": self.density_parametric_form,
            "dropout_rate": self.dropout_rate,
            "density_layer_sizes": self.density_layer_sizes,
            "decision_layer_sizes": self.decision_layer_sizes,
            "density_weight_param": self.density_weight_param,
            "decision_weight_param": self.decision_weight_param,
            "weight_penalty_type": self.weight_penalty_type,
            "cost_decline": self.cost_decline,
            "do_no_harm_param": self.do_no_harm_param,
            "log_barrier_param": self.log_barrier_param,
            "max_iters": self.max_iters,
            "num_inits": self.num_inits,
            "num_ensemble": self.num_ensemble,
            "act_func": self.act_func,
            "learning_rate": self.learning_rate,
            "support_sim_settings": self.support_sim_settings,
            "support_sim_num": self.support_sim_num,
        }

    def set_params(self, **params):
        if "density_parametric_form" in params:
            self.density_parametric_form = params["density_parametric_form"]
        if "dropout_rate" in params:
            self.dropout_rate = params["dropout_rate"]
        if "density_layer_sizes" in params:
            self.density_layer_sizes = params["density_layer_sizes"]
        if "decision_layer_sizes" in params:
            self.decision_layer_sizes = params["decision_layer_sizes"]
        if "decision_weight_param" in params:
            self.decision_weight_param = params["decision_weight_param"]
        if "density_weight_param" in params:
            self.density_weight_param = params["density_weight_param"]
        if "weight_penalty_type" in params:
            self.weight_penalty_type = params["weight_penalty_type"]
        if "cost_decline" in params:
            self.cost_decline = params["cost_decline"]
        if "do_no_harm_param" in params:
            self.do_no_harm_param = params["do_no_harm_param"]
        if "log_barrier_param" in params:
            self.log_barrier_param = params["log_barrier_param"]
        if "act_func" in params:
            self.act_func = params["act_func"]
        if "max_iters" in params:
            self.max_iters = params["max_iters"]
        if "num_inits" in params:
            self.num_inits = params["num_inits"]
        if "num_ensemble" in params:
            self.num_ensemble = params["num_ensemble"]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "support_sim_settings" in params:
            self.support_sim_settings = params["support_sim_settings"]
        if "support_sim_num" in params:
            self.support_sim_num = params["support_sim_num"]
        self._init_nn()

    def get_accept_prob(self, x, max_replicates: int=100):
        if len(self.nns) == 1:
            return self.nns[0].get_accept_prob(x, max_replicates)
        elif self.decision_layer_sizes is not None:
            accept_probs = 0
            for nn in self.nns:
                accept_probs += nn.get_accept_prob(x, max_replicates)
            return accept_probs/len(self.nns)
        else:
            entropies = self.get_entropy(x, max_replicates)
            return np.array(entropies < self.cost_decline, dtype=int)

    def get_entropy(self, x, max_replicates: int = 100):
        """
        @param x: covariates
        @return accept prob (return N x 1 vector)
        """
        cond_dist = self.get_cond_dists(x, max_replicates)
        if self.density_parametric_form == "gaussian":
            return cond_dist.entropy().reshape((-1,1))
        elif self.density_parametric_form == "bernoulli":
            cond_dist = cond_dist * 0.999 + (1 - 0.999)/2
            return -np.sum(cond_dist * np.log(cond_dist) + (1 - cond_dist) * np.log(1 - cond_dist), axis=1, keepdims=True)
        else:
            return np.sum(-cond_dist * np.log(cond_dist), axis=1, keepdims=True)

    def get_density(self, x, y, max_replicates=100):
        """
        @param x: covariates
        @param y: response
        @return density of Y|X
        """
        cond_dist = self.get_cond_dists(x, max_replicates)
        if self.density_parametric_form == "gaussian":
            return cond_dist.pdf(y).reshape((-1,1))
        elif self.density_parametric_form == "bernoulli":
            p_y_x = np.sum(cond_dist * y + (1 - cond_dist) * (1 - y), axis=1, keepdims=True)
            return p_y_x * 0.999 + (1 - 0.999)/2
        else:
            return np.sum(cond_dist * y, axis=1, keepdims=True)

    def get_prediction_interval(self, x, alpha=0.1, max_replicates=100):
        cond_dist = self.get_cond_dists(x, max_replicates)
        if self.density_parametric_form == "gaussian":
            lower_quantile = np.ones((x.shape[0], 1)) * alpha/2
            upper_quantile = np.ones((x.shape[0], 1)) * (1 - alpha/2)
            lower_pi = cond_dist.ppf(lower_quantile)
            upper_pi = cond_dist.ppf(upper_quantile)
            return np.concatenate([lower_pi, upper_pi], axis=1)
        elif self.density_parametric_form == "bernoulli":
            return get_bernoulli_pred_interval(cond_dist, alpha)
        else:
            return get_multinomial_pred_interval(cond_dist, alpha)

    def get_cond_dists(self, x, max_replicates=100):
        num_replicates = max_replicates if self.dropout_rate > 0 else 1
        sess = tf.Session()
        with sess.as_default():
            cond_dists = []
            if self.density_parametric_form == "gaussian":
                # USE APPROXIMATE FOR NOW
                print("THIS IS AN APPROXIMATE MIXTURE OF GAUSSIANS")
                final_mus, final_sigmas = [], []
                for model in self.nns:
                    model._init_network_variables(sess)
                    all_mus = []
                    all_sigmas = []
                    for i in range(num_replicates):
                        mu, sigma = sess.run(
                            [model.mu, model.sigma],
                            feed_dict={
                                model.x_concat_ph: x,
                            })
                        all_mus.append(mu)
                        all_sigmas.append(sigma)
                    final_mu, final_sigma = get_mu_sigma_of_mixture_normals(all_mus, all_sigmas)
                    final_mus.append(final_mu)
                    final_sigmas.append(final_sigma)
                final_final_mu, final_final_sigma = get_mu_sigma_of_mixture_normals(final_mus, final_sigmas)
                cond_dist = scipy.stats.norm(loc=final_final_mu, scale=final_final_sigma)
                cond_dists.append(cond_dist)
            else:
                all_ps = 0
                for model in self.nns:
                    model._init_network_variables(sess)
                    mu = sess.run(
                        model.mu,
                        feed_dict={
                            model.x_concat_ph: x})
                    all_ps += mu
                cond_dist = all_ps/len(self.nns)
            cond_dists.append(cond_dist)
        return cond_dist

    def get_prediction_probs(self, x, max_replicates=100):
        assert self.density_parametric_form != "gaussian"
        return self.get_cond_dists(x, max_replicates)

    def score(self, x, y, max_replicates=500):
        accept_prob = self.get_accept_prob(x, max_replicates)
        neg_ll = -np.log(self.get_density(x,y, max_replicates))
        return -np.mean(neg_ll * accept_prob + self.cost_decline * (1 - accept_prob))

    @property
    def has_density(self):
        return True
