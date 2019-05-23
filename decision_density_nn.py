import time
import logging

import tensorflow as tf
import numpy as np
from numpy import ndarray

from support_sim_settings import SupportSimSettings
from decision_density_base_nn import DensityDecisionBaseNN

class SimultaneousDensityDecisionNNs(DensityDecisionBaseNN):
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
            max_iters=200,
            act_func: str="tanh",
            learning_rate: float=0.002,
            sgd_batch_size: int=4000,
            support_sim_settings: SupportSimSettings=None,
            support_sim_num: int=100,
            model_params=None):
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
        self.prediction_weight_param = density_weight_param
        self.weight_penalty_type = weight_penalty_type
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
        self.model_params = model_params

    def get_params(self, deep=True):
        return {
            "density_parametric_form": self.density_parametric_form,
            "dropout_rate": self.dropout_rate,
            "density_layer_sizes": self.density_layer_sizes,
            "decision_layer_sizes": self.decision_layer_sizes,
            "density_weight_param": self.prediction_weight_param,
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
            self.prediction_weight_param = params["density_weight_param"]
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
