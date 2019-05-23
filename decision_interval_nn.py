import time
import logging

import tensorflow as tf
import numpy as np

from neural_network_wrapper import NeuralNetworkParams
from neural_network import NeuralNetwork
from support_sim_settings import SupportSimSettings
from decision_interval_base_nn import IntervalDecisionBaseNN

class SimultaneousIntervalDecisionNNs(IntervalDecisionBaseNN):
    """
    Fits a `decision_nn` that decides whether or not to accept
    Fits a `interval_nn` for the prediction interval
    """
    def __init__(self,
            decision_layer_sizes=None,
            interval_layer_sizes=None,
            decision_weight_param=0.01,
            interval_weight_param=0.01,
            shared_layer=0,
            dropout_rate=0,
            weight_penalty_type="ridge",
            interval_alpha=0.05,
            cost_decline=0.01,
            do_no_harm_param=0.01,
            log_barrier_param=0.01,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            learning_rate=0.002,
            sgd_batch_size=4000,
            support_sim_settings: SupportSimSettings=None,
            support_sim_num=100):
        """
        (other parameters defined in fit_simultaneous_decision_interval_nn.py)
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param learning_rate: learning rate for adam optimization
        @param support_sim_settings: SupposeSimSettings -- specifies how to simulate
                    points over the support of X
        @param support_sim_num: number of points to simulate at each step
                    of SGD to approximate the integral
        """
        self.decision_layer_sizes = decision_layer_sizes
        self.interval_layer_sizes = interval_layer_sizes
        self.prediction_layer_sizes = interval_layer_sizes
        self.shared_layer = shared_layer
        self.dropout_rate = dropout_rate
        self.decision_weight_param = decision_weight_param
        self.prediction_weight_param = interval_weight_param
        self.weight_penalty_type = weight_penalty_type
        self.interval_alpha = interval_alpha
        self.cost_decline = cost_decline
        self.do_no_harm_param = do_no_harm_param
        self.log_barrier_param = log_barrier_param
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.sgd_batch_size = sgd_batch_size
        self.support_sim_settings = support_sim_settings
        self.support_sim_num = support_sim_num
        if self.interval_layer_sizes:
            self._init_nn()

    def get_params(self, deep=True):
        return {
            "decision_layer_sizes": self.decision_layer_sizes,
            "interval_layer_sizes": self.interval_layer_sizes,
            "shared_layer": self.shared_layer,
            "dropout_rate": self.dropout_rate,
            "interval_weight_param": self.prediction_weight_param,
            "decision_weight_param": self.decision_weight_param,
            "weight_penalty_type": self.weight_penalty_type,
            "cost_decline": self.cost_decline,
            "do_no_harm_param": self.do_no_harm_param,
            "log_barrier_param": self.log_barrier_param,
            "max_iters": self.max_iters,
            "num_inits": self.num_inits,
            "act_func": self.act_func,
            "learning_rate": self.learning_rate,
            "support_sim_settings": self.support_sim_settings,
            "support_sim_num": self.support_sim_num,
            "interval_alpha": self.interval_alpha,
        }

    def set_params(self, **params):
        if "decision_layer_sizes" in params:
            self.decision_layer_sizes = params["decision_layer_sizes"]
        if "interval_layer_sizes" in params:
            self.interval_layer_sizes = params["interval_layer_sizes"]
        if "decision_weight_param" in params:
            self.decision_weight_param = params["decision_weight_param"]
        if "shared_layer" in params:
            self.shared_layer = params["shared_layer"]
        if "dropout_rate" in params:
            self.dropout_rate = params["dropout_rate"]
        if "interval_alpha" in params:
            self.prediction_alpha = params["interval_alpha"]
        if "interval_weight_param" in params:
            self.prediction_weight_param = params["interval_weight_param"]
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
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "support_sim_settings" in params:
            self.support_sim_settings = params["support_sim_settings"]
        if "support_sim_num" in params:
            self.support_sim_num = params["support_sim_num"]
        self._init_nn()
