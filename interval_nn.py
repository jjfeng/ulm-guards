import time
import logging

import tensorflow as tf
import numpy as np
from numpy import ndarray

from prediction_nn import PredictionNN
from neural_network import NeuralNetwork

class IntervalNN(PredictionNN):
    def __init__(self,
            interval_layer_sizes=None,
            interval_weight_param=0.01,
            weight_penalty_type="ridge",
            interval_alpha=0.05,
            dropout_rate=0,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            learning_rate=0.002,
            model_params=None,
            sgd_batch_size=4000):
        self.interval_alpha = interval_alpha
        self.interval_layer_sizes = interval_layer_sizes
        self.prediction_weight_param = interval_weight_param
        self.weight_penalty_type = weight_penalty_type
        self.dropout_rate = dropout_rate
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.model_params = model_params
        self.sgd_batch_size = sgd_batch_size

    def _init_nn(self, init_inner=None):
        """
        Creates the density NN
        Creates the other values needed for the objective function
        Creates the optimizers of the objective function

        @param init_inner: ignored
        """
        self.label_size = 1
        self.num_p = int(self.interval_layer_sizes[0])
        self.prediction_layer_sizes = self.interval_layer_sizes

        # Input layers; create placeholders in tensorflow
        self.x_concat_ph = tf.placeholder(tf.float32, [None, self.num_p], name="x")
        self.y_concat_ph = tf.placeholder(tf.float32, [None, self.label_size], name="y")
        self.prediction_nn = self._create_interval_nn()

        self.prediction_loss_obs = self.interval_alpha * self.radius + tf.nn.relu(self.lower_bound - self.y_concat_ph) + tf.nn.relu(self.y_concat_ph - self.upper_bound)
        self.decision_univar_mapping = self.radius * self.interval_alpha
        self._setup_optimization_problem()

    def _create_interval_nn(self, nonzero_boost=1e-5):
        """
        Creates the neural network for the interval NN
        """
        self.interval_nn = NeuralNetwork.create_full_nnet(
            self.interval_layer_sizes + ["2"],
            self.x_concat_ph,
            act_func=getattr(tf.nn, self.act_func),
            output_act_func=None,
            dropout_rate=self.dropout_rate)
        self.mu = self.interval_nn.layers[-1][:, 0:1]
        self.radius = tf.abs(self.interval_nn.layers[-1][:, 1:2])
        self.lower_bound = self.mu - self.radius
        self.upper_bound = self.mu + self.radius
        return self.interval_nn
