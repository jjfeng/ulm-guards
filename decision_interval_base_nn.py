import time
import logging

import tensorflow as tf
import numpy as np
from numpy import ndarray

from neural_network import NeuralNetwork
from decision_prediction_nn import DecisionPredictionNNs

class IntervalDecisionBaseNN(DecisionPredictionNNs):
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

    def _init_nn(self, init_inner=None):
        """
        Handles the bulk of the logic
        Creates the decision and interval NN
        Creates the other values needed for the objective function
        Creates the optimizers of the objective function
        """
        self.label_size = 1
        self.num_p = int(self.interval_layer_sizes[0])
        self.prediction_layer_sizes = self.interval_layer_sizes

        self._setup_skeleton()
        self.prediction_nn = self._create_interval_nn()

        self.prediction_loss_obs = self.interval_alpha * self.radius + tf.nn.relu(self.lower_bound - self.y_concat_ph) + tf.nn.relu(self.y_concat_ph - self.upper_bound)
        self.decision_univar_mapping = self.radius * self.interval_alpha
        self._setup_optimization_problem()

    def _print_train(self, sess, x_train, y_train):
        """
        Utility function for printing things regarding the model (and training data) during training
        """
        accept_probs, mu, radius = sess.run(
            [
                self.accept_prob,
                self.mu,
                self.radius],
            feed_dict={
                self.x_concat_ph: x_train,
                self.y_concat_ph: y_train,
                self.is_data_ph: np.ones((x_train.shape[0], 1)),
            })
        tot_prob = np.sum(accept_probs)
        logging.info(
                "  mean mu %f, mean radius %f",
                np.sum(accept_probs * mu)/tot_prob,
                np.sum(accept_probs * radius)/tot_prob)

    def get_prediction_interval(self, x, alpha=0.1):
        """
        @param x: covariates
        @param alpha: create PI with width (1 - alpha) coverage
        @return prediction interval
        """
        if not np.isclose(alpha, self.interval_alpha):
            print(alpha, self.interval_alpha)
            raise ValueError("This NN did not train on the requested alpha")

        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            # Predict values
            lower_pi, upper_pi = sess.run(
                [self.lower_bound, self.upper_bound],
                feed_dict={
                    self.x_concat_ph: x,
                })

        sess.close()
        return np.concatenate([lower_pi, upper_pi], axis=1)

    def get_interval_length(self, x, alpha=0.1):
        pred_int = self.get_prediction_interval(x, alpha)
        return pred_int[:,1] - pred[:,0]

    @property
    def has_density(self):
        return False
