import time
import logging

import tensorflow as tf
import numpy as np
from numpy import ndarray

from prediction_nn import PredictionNN
from neural_network import NeuralNetwork

class DensityNN(PredictionNN):
    def __init__(self,
            density_layer_sizes=None,
            density_parametric_form="gaussian",
            density_weight_param=0.01,
            dropout_rate=0,
            weight_penalty_type="ridge",
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            learning_rate=0.002,
            model_params=None,
            sgd_batch_size=4000):
        self.density_parametric_form = density_parametric_form
        self.density_layer_sizes = density_layer_sizes
        self.prediction_weight_param = density_weight_param
        self.dropout_rate = dropout_rate
        self.weight_penalty_type = weight_penalty_type
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.model_params = model_params
        self.sgd_batch_size = sgd_batch_size
        #self._init_nn()

    def _init_nn(self):
        """
        Creates the density NN
        Creates the other values needed for the objective function
        Creates the optimizers of the objective function
        """
        self.label_size = 1
        if self.density_parametric_form.startswith("multinomial"):
            self.label_size = int(self.density_parametric_form.replace("multinomial", ""))
        self.num_p = self.density_layer_sizes[0]
        self.prediction_layer_sizes = self.density_layer_sizes

        # Input layers; create placeholders in tensorflow
        self.x_concat_ph = tf.placeholder(tf.float32, [None, self.num_p], name="x")
        self.y_concat_ph = tf.placeholder(tf.float32, [None, self.label_size], name="y")
        self.prediction_nn = self._create_density_nn()
        self.prediction_loss_obs = -self.log_prob
        self.decision_univar_mapping = self._get_decision_univar_mapping()
        self._setup_optimization_problem()

    def _get_decision_univar_mapping(self):
        """
        We will use entropy
        """
        if self.density_parametric_form in ["gaussian", "bernoulli"]:
            entropy = self.cond_dist.entropy()
        else:
            p = (self.mu + 1e-10)/(1 + 1e-10 * self.label_size)
            entropy = tf.reduce_sum(-p * tf.log(p), axis=1, keepdims=True)
        return entropy

    def _create_density_nn(self, nonzero_boost=1e-5):
        """
        Creates the neural network for the density NN
        """
        self.quantile_ph = tf.placeholder(tf.float32, [None, 1])
        if self.density_parametric_form == "gaussian":
            sigma_offset = tf.Variable([0.0])
            self.density_nn = NeuralNetwork.create_full_nnet(
                self.density_layer_sizes + ["2"],
                self.x_concat_ph,
                act_func=getattr(tf.nn, self.act_func),
                output_act_func=None,
                dropout_rate=self.dropout_rate)
            self.mu = self.density_nn.layers[-1][:, 0:1]
            # Cannot let sigma get too close to zero -- allow an offset parameter
            self.sigma = tf.abs(self.density_nn.layers[-1][:, 1:2]) + tf.exp(sigma_offset)
            # Add offset parameter to the density NN intercepts so we will train over its value
            self.density_nn.intercepts += [sigma_offset]
            self.cond_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
            self.quantile = self.cond_dist.quantile(self.quantile_ph)
            self.log_prob = self.cond_dist.log_prob(self.y_concat_ph)
        elif self.density_parametric_form.startswith("shifted_exponential"):
            raise ValueError("doesn't work well in tensorflow for some stupid reason")
            rate_offset = tf.Variable(-0.5)
            #raise ValueError("This doesnt work for some weird reason. think tensorflow has issues")
            self.min_y = float(self.density_parametric_form.replace("shifted_exponential", ""))
            self.density_nn = NeuralNetwork.create_full_nnet(
                self.density_layer_sizes + ["1"],
                self.x_concat_ph,
                act_func=getattr(tf.nn, self.act_func),
                output_act_func=None,
                dropout_rate=self.dropout_rate)
            self.rate = tf.exp(tf.minimum(self.density_nn.layers[-1][:,0], 2.5)) + tf.exp(rate_offset)
            self.density_nn.intercepts += [rate_offset]
            # Add offset parameter to the density NN intercepts so we will train over its value
            self.cond_dist = tf.distributions.Exponential(rate=self.rate, validate_args=True)
            self.log_prob = self.cond_dist.log_prob(self.y_concat_ph - self.min_y)
            #self.log_prob = self.cond_dist.log_prob(self.y_concat_ph)
        elif self.density_parametric_form == "bernoulli":
            self.density_nn = NeuralNetwork.create_full_nnet(
                self.density_layer_sizes + ["1"],
                self.x_concat_ph,
                act_func=getattr(tf.nn, self.act_func),
                output_act_func=None,
                dropout_rate=self.dropout_rate)
            self.logits = self.density_nn.layers[-1]
            self.mu = tf.nn.sigmoid(self.logits)
            self.cond_dist = tf.distributions.Bernoulli(logits=self.logits)
            self.log_prob = self.cond_dist.log_prob(self.y_concat_ph)
        elif self.density_parametric_form.startswith("multinomial"):
            self.density_nn = NeuralNetwork.create_full_nnet(
                self.density_layer_sizes + [str(self.label_size)],
                self.x_concat_ph,
                act_func=getattr(tf.nn, self.act_func),
                output_act_func=None,
                dropout_rate=self.dropout_rate)
            self.logits = self.density_nn.layers[-1]
            self.cond_dist = tf.distributions.Multinomial(total_count=1., logits=self.logits)
            self.mu = self.cond_dist.mean()
            # Weird.... it returns something the wrong shape. -- we do reshape to fix it
            self.log_prob = tf.reshape(self.cond_dist.log_prob(self.y_concat_ph), (-1,1))
            #self.log_prob = tf.log(tf.reduce_sum(self.mu * self.y_concat_ph, axis=1, keepdims=True))
        else:
            raise ValueError("Dont know about this form")
        return self.density_nn
