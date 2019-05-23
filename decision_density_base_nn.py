import time
import logging

import tensorflow as tf
import numpy as np
import scipy.stats
from numpy import ndarray

from neural_network import NeuralNetwork
from decision_prediction_nn import DecisionPredictionNNs
from common import *

class DensityDecisionBaseNN(DecisionPredictionNNs):
    def _create_density_nn(self, nonzero_boost=1e-5):
        """
        Creates the neural network for the density NN
        """
        self.quantile_ph = tf.placeholder(tf.float32, [None, 1])
        if self.density_parametric_form == "gaussian":
            sigma_offset = tf.Variable(0.0)
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

    def _init_nn(self, init_inner=None):
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

        self._setup_skeleton()
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

    def get_density(self, x, y, max_replicates=1000):
        """
        @param x: covariates
        @param y: response
        @return density of Y|X
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            # Predict values
            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            all_densities = []
            for i in range(num_replicates):
                log_density = sess.run(self.log_prob,
                    feed_dict={
                        self.x_concat_ph: x,
                        self.y_concat_ph: y,
                    })
                all_densities.append(np.exp(log_density))

        sess.close()
        final_densities = np.mean(np.concatenate(all_densities, axis=1), axis=1)
        return final_densities

    def get_prediction_interval(self, x: ndarray, alpha=0.1, max_replicates=1000):
        """
        @param x: covariates
        @param alpha: create PI with width (1 - alpha) coverage
        @return prediction intervals
        """
        assert x.size > 0
        assert alpha > 0 and alpha < 1
        num_replicates = max_replicates if self.dropout_rate > 0 else 1
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            if self.density_parametric_form == "gaussian":
                prediction_interval = self._get_gaussian_prediction_interval(x, alpha, sess, num_replicates)
            elif self.density_parametric_form == "bernoulli":
                prediction_interval = self._get_bernoulli_prediction_interval(x, alpha, sess, num_replicates)
            else:
                prediction_interval = self._get_multinomial_prediction_interval(x, alpha, sess, num_replicates)

        sess.close()
        return prediction_interval

    def _get_gaussian_prediction_interval(self, x: ndarray, alpha: float, sess, num_replicates: int = 1):
        """
        @return a (1-alpha) prediction interval
        """
        lower_quantile = np.ones((x.shape[0], 1)) * alpha/2
        upper_quantile = np.ones((x.shape[0], 1)) * (1 - alpha/2)

        all_mus = []
        all_sigmas = []
        for i in range(num_replicates):
            mu, sigma = sess.run(
                [self.mu, self.sigma],
                feed_dict={
                    self.x_concat_ph: x,
                })
            all_mus.append(mu)
            all_sigmas.append(sigma)
        final_mu, final_sigma = get_mu_sigma_of_mixture_normals(all_mus, all_sigmas)

        lower_pi = scipy.stats.norm.ppf(lower_quantile, loc=final_mu, scale=final_sigma)
        upper_pi = scipy.stats.norm.ppf(upper_quantile, loc=final_mu, scale=final_sigma)

        # Predict values
        #lower_pi1 = sess.run(
        #    self.quantile,
        #    feed_dict={
        #        self.x_concat_ph: x,
        #        self.quantile_ph: lower_quantile,
        #    })
        #upper_pi1 = sess.run(
        #    self.quantile,
        #    feed_dict={
        #        self.x_concat_ph: x,
        #        self.quantile_ph: upper_quantile,
        #    })

        return np.concatenate([lower_pi, upper_pi], axis=1)

    def _get_bernoulli_prediction_interval(self, x: ndarray, alpha: float, sess, num_replicates=1):
        """
        @return a prediction INTERVAL with probability at least 1 - alpha
                (e.g. [0,1] means both 0 and 1 are possible)

        TODO: unify this with the multinomial version?
        """
        predicted_mu = 0
        for i in range(num_replicates):
            predicted_mu_rep = sess.run(
                self.mu,
                feed_dict={
                    self.x_concat_ph: x,
                })
            predicted_mu += predicted_mu_rep
        predicted_mu = predicted_mu/num_replicates
        return get_bernoulli_pred_interval(predicted_mu, alpha)

    def _get_multinomial_prediction_interval(self, x: ndarray, alpha: float, sess, num_replicates=1):
        """
        A prediction interval for a multiclass with K classes is a K-dim vector:
           e.g. [0, 1, 0, 1] means class 1 and 3 are in the prediction set
        so this is closer to a set
        """
        predicted_mu = 0
        for i in range(num_replicates):
            predicted_mu_rep = sess.run(
                self.mu,
                feed_dict={
                    self.x_concat_ph: x,
                })
            predicted_mu += predicted_mu_rep
        predicted_mu = predicted_mu/num_replicates

        return get_multinomial_pred_interval(predicted_mu, alpha)

    def get_prediction_probs(self, x: ndarray, max_replicates: int = 1000):
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            num_replicates = max_replicates if self.dropout_rate > 0 else 1
            predicted_mu = 0
            for i in range(num_replicates):
                predicted_mu_rep = sess.run(
                    self.mu,
                    feed_dict={
                        self.x_concat_ph: x,
                    })
                predicted_mu += predicted_mu_rep
            predicted_mu = predicted_mu/num_replicates

        sess.close()
        return predicted_mu

    def get_prediction_dist(self, x: ndarray, max_replicates: int = 1000):
        """
        @param x: covariates
        @return predicted mean and standard deviation of y|x for Gaussian model
        """
        assert self.density_parametric_form == "gaussian"
        sess = tf.Session()
        num_replicates = max_replicates if self.dropout_rate > 0 else 1
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            all_mus = []
            all_sigmas = []
            for i in range(num_replicates):
                mu, sigma = sess.run(
                    [self.mu, self.sigma],
                    feed_dict={
                        self.x_concat_ph: x,
                    })
                all_mus.append(mu)
                all_sigmas.append(sigma)
            final_mu, final_sigma = get_mu_sigma_of_mixture_normals(all_mus, all_sigmas)

        sess.close()
        return np.concatenate([mu, sigma], axis=1)

    def get_entropy(self, x: ndarray, max_replicates: int=1000):
        if self.density_parametric_form == "gaussian":
            sigma = self.get_prediction_dist(x, max_replicates)[:,1]
            return get_normal_dist_entropy(sigma)
        elif self.density_parametric_form == "bernoulli":
            p = self.get_prediction_probs(x, max_replicates)
            return -p * np.log(p) - (1 - p) * np.log(1 - p)
        else:
            p = self.get_prediction_probs(x, max_replicates)
            p = (p + 1e-10)/(1 + 1e-10 * self.label_size)
            return np.sum(-p * np.log(p), axis=1)

    @property
    def has_density(self):
        return True
