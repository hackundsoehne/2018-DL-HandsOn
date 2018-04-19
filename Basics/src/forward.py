import numpy as np
from math import fabs, fsum

class ForewardWeights:
    """
    This class provides the weights for the forward-propagation excersice.
    """
    def __init__(self):
        self.weights = None
    
    def getWeights(self, layer, neuron):
        """
        returns the weights for an individual neuron
        """
        if (layer < 0 | layer > 2):
            raise ValueError("there are only 3 layers: 2,1,0")
        return weights[layer][neuron]
        
        
class Evaluation:
    """
    Evaluates the a neural network for the forward-propagation excersice.
    """
    def __init__(self):
        self.test_x = None
        self.test_y = None
        
    def get_test(self):
        return test_x
    
    def eval(self, results):
        """
        Returns the mean absolute error (MAE) of the neural network for the prediction.
        """
        losses = [fabs(ni - yi) for ni in results for yi in test_y]
        return fsum(losses)/length(losses)