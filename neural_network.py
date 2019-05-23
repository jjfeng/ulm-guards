import time
import logging as log

import tensorflow as tf
import numpy as np

from sklearn.base import BaseEstimator
from common import process_params, mult_list

class NeuralNetwork(BaseEstimator):
    """
    Super class for neural nets.
    Has functionality for making ordinary nnets.
    """
    def __init__(self, coefs, intercepts, layers):
        """
        initialization for fully connected, standard nnets

        @param layers: list of nodes at each layer
        """
        self.coefs = coefs
        self.intercepts = intercepts
        self.var_list = coefs + intercepts
        self.layers = layers

    @staticmethod
    def get_init_rand_bound_tanh(shape):
        # Used for tanh
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(6. / np.sum(shape))

    @staticmethod
    def get_init_rand_bound_sigmoid(shape):
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(2. / np.sum(shape))

    @staticmethod
    def create_tf_var(shape):
        bound = NeuralNetwork.get_init_rand_bound_tanh(shape)
        return tf.Variable(tf.random_uniform(shape, minval=-bound, maxval=bound))

    @staticmethod
    def create_full_nnet(layer_sizes, input_layer, act_func=tf.nn.tanh, output_act_func=None, dropout_rate=0):
        """
        @param input_layer: input layer (tensor)
        @param layer_sizes: size of each layer (input to output)
        """
        coefs = []
        intercepts = []
        layers = []
        n_layers = len(layer_sizes)
        print(layer_sizes)
        for i in range(n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            #if fan_out.startswith("conv"):
            #    # We're doing a convolutional layer...
            #    W_size, strides = fan_out.replace("conv", "").split(":")
            #    strides = process_params(strides, int, split_str="~")
            #    W_size = process_params(W_size, int, split_str="~")
            #    b_size = [W_size[-1]]
            #    W = NeuralNetwork.create_tf_var(W_size)
            #    b = NeuralNetwork.create_tf_var(b_size)
            #    layer = tf.nn.conv2d(input_layer, W, strides=strides, padding='SAME') + b
            #elif fan_in.startswith("conv") and not fan_out.startswith("conv"):
            #    # Transitioning to non-conv from a conv layer
            #    fan_in = int(mult_list(input_layer.shape[1:]))
            #    print("flatten size", fan_in)
            #    fan_out = int(fan_out)
            #    input_layer = tf.reshape(input_layer, [-1, fan_in])
            #    W_size = [fan_in, fan_out]
            #    b_size = [1,fan_out]
            #    W = NeuralNetwork.create_tf_var(W_size)
            #    b = NeuralNetwork.create_tf_var(b_size)
            #    layer = tf.add(tf.matmul(input_layer, W), b)
            if dropout_rate > 0:
                input_layer = tf.nn.dropout(input_layer, keep_prob=1 - dropout_rate)
            fan_in = int(fan_in)
            fan_out = int(fan_out)
            W_size = [fan_in, fan_out]
            b_size = [1,fan_out]
            W = NeuralNetwork.create_tf_var(W_size)
            b = NeuralNetwork.create_tf_var(b_size)
            layer = tf.add(tf.matmul(input_layer, W), b)
            if i < n_layers - 2:
                # if not last layer, add activation
                layer = act_func(layer)
            else:
                # is the layer layer
                if output_act_func is not None:
                    layer = output_act_func(layer)
            input_layer = layer
            coefs.append(W)
            intercepts.append(b)
            layers.append(layer)

        return NeuralNetwork(coefs, intercepts, layers)

    @staticmethod
    def create_full_nnet_dense(layer_sizes, input_layer, act_func=tf.nn.tanh, output_act_func=None):
        """
        @param input_layer: input layer (tensor)
        @param layer_sizes: size of each layer (input to output)
        """
        coefs = []
        intercepts = []
        layers = []
        n_layers = len(layer_sizes)
        for i in range(n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W_size = [fan_in, fan_out]
            b_size = [1,fan_out]
            W = NeuralNetwork.create_tf_var(W_size)
            b = NeuralNetwork.create_tf_var(b_size)
            layer = tf.add(tf.matmul(input_layer, W), b)
            if i < n_layers - 2:
                # if not last layer, add activation
                layer = act_func(layer)
            else:
                # is the layer layer
                if output_act_func is not None:
                    layer = output_act_func(layer)
            input_layer = layer
            coefs.append(W)
            intercepts.append(b)
            layers.append(layer)

        return NeuralNetwork(coefs, intercepts, layers)
