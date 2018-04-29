import numpy as np
from math import fabs, fsum

class BackwardWeights:
    """
    This class provides the weights for the forward-propagation excersice
    """
    def __init__(self):
        self.weights = None
        self.biases = None
        
    def updateWeights(self, layer, neuron, weights):
        """
        assigns new weights for an individual neuron
        """
        return weights[layer][neuron] = weights
    
    def getWeights(self, layer, neuron):
        """
        returns the weights for an individual neuron
        """
        if (layer < 0 | layer > 2):
            raise ValueError("there are only 3 layers: 2,1,0")
        return weights[layer][neuron]
    
    def getBias(self, layer, neuron):
        """
        returns the weights for an individual neuron
        """
        if (layer < 0 | layer > 2):
            raise ValueError("there are only 3 layers: 2,1,0")
        return biases[layer][neuron]
    
    def updateBias(self, layer, neuron, bias):
        """
        assigns new weights for an individual neuron
        """
        if (layer < 0 | layer > 2):
            raise ValueError("there are only 3 layers: 2,1,0")
        return biases[layer][neuron] = bias
        
        
class Train:
    """
    Evaluates the a neural network for the forward-propagation excersice.
    """
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        
    def get_test(self):
        return test_x
    
    def train(self, trainFkt, forewardFkt):
        for (i in range (1,3000)):
            i = rand.uniform(0, train_x.shape()[0])
            trainFkt(train_x[i], train[i])
            if (i % 20 == 0):
                //TODO error
        //TODO total error
    
    def eval(self, results):
        """
        Returns the mean absolute error (MAE) of the neural network for the prediction.
        """
        losses = [fabs(ni - yi) for ni in results for yi in test_y]
        return fsum(losses)/length(losses)
    
class TrainHelper:
    """
    TrainHelper for the assignments
    """
    def __init__(self):
        sef.train = new Train()
        
    def 