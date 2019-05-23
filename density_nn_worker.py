import tensorflow as tf

from parallel_worker import ParallelWorker
from neural_network import NeuralNetwork
from decision_prediction_nn import DecisionPredictionNNs
from common import get_normal_dist_entropy, get_mu_sigma_of_mixture_normals

class NNWorker(ParallelWorker):
    def __init__(
        self,
        seed,
        neural_net):
        """
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param learning_rate: learning rate for adam optimization
        """
        self.seed = seed
        self.nn = nn
