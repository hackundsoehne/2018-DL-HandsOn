import numpy as np

class TrainEval:
    """
    Evaluates the a neural network on the XX dataset.
    """
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
            
    
    def train():
        raise ValueError("not implemented yet")
    
    def eval(neural_network, loss):
        res = [neural_network(xi) for xi in test_x]
        losses = [loss(ni, yi) for ni in res, yi in test_y]
        return 
        