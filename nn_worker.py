import tensorflow as tf

from parallel_worker import ParallelWorker
from neural_network_wrapper import NeuralNetworkParams

class NNWorker(ParallelWorker):
    def __init__(
            self,
            seed,
            neural_net):
        """
        """
        self.seed = seed
        self.nn = neural_net

    def run_worker(self, shared_obj):
        """
        """
        self.nn._init_nn()
        X = shared_obj[0]
        y = shared_obj[1]
        self.nn.fit(X, y)
        return self.nn.model_params
