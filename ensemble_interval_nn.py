import time
import logging
import copy

import tensorflow as tf
import numpy as np
import scipy.stats
from numpy import ndarray

from interval_nn import IntervalNN
from ensemble_prediction_nn import EnsemblePredictionNN

class EnsembleIntervalNN(EnsemblePredictionNN):
    def __init__(self,
            interval_layer_sizes=None,
            interval_weight_param=0.01,
            weight_penalty_type="ridge",
            interval_alpha=0.05,
            dropout_rate=0,
            num_ensemble=1,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            learning_rate=0.002,
            sgd_batch_size=4000,
            do_distributed=False,
            scratch_dir="/fh/fast/matsen_e/jfeng2/scratch"):
        """
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param learning_rate: learning rate for adam optimization
        """
        # Right now, we only do a single NN for interval estimation
        self.num_ensemble = 1

        self.interval_alpha = interval_alpha
        self.interval_layer_sizes = interval_layer_sizes
        self.interval_weight_param = interval_weight_param
        self.weight_penalty_type = weight_penalty_type
        self.dropout_rate = dropout_rate
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.sgd_batch_size = sgd_batch_size
        self.do_distributed = do_distributed
        self.scratch_dir = scratch_dir
        self._init_nn()

    def _init_nn(self, init_inner=False):
        self.nns = [IntervalNN(
            interval_layer_sizes=self.interval_layer_sizes,
            interval_weight_param=self.interval_weight_param,
            weight_penalty_type=self.weight_penalty_type,
            interval_alpha=self.interval_alpha,
            dropout_rate=self.dropout_rate,
            num_inits=self.num_inits,
            max_iters=self.max_iters,
            act_func=self.act_func,
            learning_rate=self.learning_rate,
            model_params=None,
            sgd_batch_size=self.sgd_batch_size)
        ]
        self.get_univar_mapping = self.nns[0].get_univar_mapping
        if init_inner:
            for nn in self.nns:
                nn._init_nn()

    def get_params(self, deep=True):
        return {
            "interval_layer_sizes": self.interval_layer_sizes,
            "interval_weight_param": self.interval_weight_param,
            "interval_alpha": self.interval_alpha,
            "dropout_rate": self.dropout_rate,
            "weight_penalty_type": self.weight_penalty_type,
            "max_iters": self.max_iters,
            "num_ensemble": self.num_ensemble,
            "num_inits": self.num_inits,
            "act_func": self.act_func,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params):
        if "interval_layer_sizes" in params:
            self.interval_layer_sizes = params["interval_layer_sizes"]
        if "interval_weight_param" in params:
            self.interval_weight_param = params["interval_weight_param"]
        if "interval_alpha" in params:
            self.interval_alpha = params["interval_alpha"]
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

    def get_accept_prob(self, x, max_replicates: int=100):
        assert len(self.nns) == 1
        return self.nns[0].get_accept_prob(x, max_replicates)
