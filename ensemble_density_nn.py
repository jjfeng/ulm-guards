import time
import logging
import copy

import tensorflow as tf
import numpy as np
import scipy.stats
from numpy import ndarray

from density_nn import DensityNN
from ensemble_prediction_nn import EnsemblePredictionNN
from common import get_mu_sigma_of_mixture_normals

class EnsembleDensityNN(EnsemblePredictionNN):
    def __init__(self,
            density_layer_sizes=None,
            density_parametric_form="gaussian",
            density_weight_param=0.01,
            dropout_rate=0,
            weight_penalty_type="ridge",
            num_ensemble=1,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            learning_rate=0.002,
            sgd_batch_size=4000,
            do_distributed=False,
            scratch_dir="scratch"):
        """
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param learning_rate: learning rate for adam optimization
        """
        self.density_parametric_form = density_parametric_form
        self.density_layer_sizes = density_layer_sizes
        self.density_weight_param = density_weight_param
        self.dropout_rate = dropout_rate
        self.weight_penalty_type = weight_penalty_type
        self.max_iters = max_iters
        self.num_ensemble = num_ensemble
        self.num_inits = num_inits
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.sgd_batch_size = sgd_batch_size
        self.do_distributed = do_distributed
        self.scratch_dir = scratch_dir
        self.get_univar_mapping = self.get_entropy
        self._init_nn()

    def _init_nn(self, init_inner=False):
        self.nns = [DensityNN(
            density_layer_sizes=self.density_layer_sizes,
            density_parametric_form=self.density_parametric_form,
            density_weight_param=self.density_weight_param,
            dropout_rate=self.dropout_rate,
            weight_penalty_type=self.weight_penalty_type,
            num_inits=self.num_inits,
            max_iters=self.max_iters,
            act_func=self.act_func,
            learning_rate=self.learning_rate,
            model_params=None,
            sgd_batch_size=self.sgd_batch_size)
            for _ in range(self.num_ensemble)
        ]
        if init_inner:
            for nn in self.nns:
                nn._init_nn()

    def get_params(self, deep=True):
        return {
            "density_parametric_form": self.density_parametric_form,
            "density_layer_sizes": self.density_layer_sizes,
            "density_weight_param": self.density_weight_param,
            "dropout_rate": self.dropout_rate,
            "weight_penalty_type": self.weight_penalty_type,
            "max_iters": self.max_iters,
            "num_ensemble": self.num_ensemble,
            "num_inits": self.num_inits,
            "act_func": self.act_func,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params):
        if "density_parametric_form" in params:
            self.density_parametric_form = params["density_parametric_form"]
        if "density_layer_sizes" in params:
            self.density_layer_sizes = params["density_layer_sizes"]
        if "density_weight_param" in params:
            self.density_weight_param = params["density_weight_param"]
        if "dropout_rate" in params:
            self.dropout_rate = params["dropout_rate"]
        if "weight_penalty_type" in params:
            self.weight_penalty_type = params["weight_penalty_type"]
        if "act_func" in params:
            self.act_func = params["act_func"]
        if "max_iters" in params:
            self.max_iters = params["max_iters"]
        if "num_ensemble" in params:
            self.num_ensemble = params["num_ensemble"]
        if "num_inits" in params:
            self.num_inits = params["num_inits"]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        self._init_nn()

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

    def get_cond_dists(self, x, max_replicates=100):
        num_replicates = max_replicates if self.dropout_rate > 0 else 1
        sess = tf.Session()
        with sess.as_default():
            cond_dists = []
            if self.density_parametric_form == "gaussian":
                #mus, sigmas = sess.run([pred_nn.mu, pred_nn.sigma],
                #    feed_dict={
                #        pred_nn.x_concat_ph: x})
                #cond_dist = scipy.stats.norm(loc=mus, scale=sigmas)
                # USE APPROXIMATE FOR NOW
                print("THIS IS AN APPROXIMATE MIXTURE OF GAUSSIANS")
                final_mus, final_sigmas = [], []
                for pred_nn in self.nns:
                    pred_nn._init_network_variables(sess)
                    all_mus = []
                    all_sigmas = []
                    for i in range(num_replicates):
                        mu, sigma = sess.run(
                            [pred_nn.mu, pred_nn.sigma],
                            feed_dict={
                                pred_nn.x_concat_ph: x,
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
                for pred_nn in self.nns:
                    pred_nn._init_network_variables(sess)
                    mu = sess.run(
                        pred_nn.mu,
                        feed_dict={
                            pred_nn.x_concat_ph: x})
                    all_ps += mu
                cond_dist = all_ps/len(self.nns)
            cond_dists.append(cond_dist)
        return cond_dist
