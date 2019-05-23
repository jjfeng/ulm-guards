class NeuralNetworkParams:
    """
    Class for storing tensorflow model parameters
    """
    def __init__(
            self,
            param_list):
        self.param_list = param_list

    def __str__(self):
        return "params %s" % self.param_list
