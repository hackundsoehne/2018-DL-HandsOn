import numpy as np
from math import fabs, fsum
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy.random as rand

class BackwardWeightsBCW:
    """
    This class provides the weights for the forward-propagation Breast Cancer Wisconsin (Diagnostic) Dataset excersice.
    """
    """
    This class provides the weights for the forward-propagation excersice.
    """
    def __init__(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
        self.weights = np.clip(np.random.normal(0, 1/np.sqrt(30), 30).reshape((1,30)), -2, 2)
        self.bias = np.clip(np.random.normal(0, 1/np.sqrt(30), 1).reshape(()), -2, 2)
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        
    def getTrainID(self, trainFkt, forewardFktProba):
        X_ID_test = np.ones(self.X_test.shape)
        X_ID_test[:,0] = self.y_test
        X_ID_train = np.ones(self.X_train.shape)
        X_ID_train[:,0] = self.y_train
        return Train(X_ID_train, self.y_train, X_ID_test, self.y_test, trainFkt, forewardFktProba, False, False)
        
    def getTrain(self, trainFkt, forewardFktProba):
        return Train(self.X_train, self.y_train, self.X_test, self.y_test, trainFkt, forewardFktProba, False, False)
    
    def setWeights(self, W):
        self.weights = W
    
    def setBias(self, b):
        self.bias = b
    
    def getWeights(self):
        """
        returns the weights
        """
        return self.weights
    
    def getBias(self):
        """
        returns the bias
        """
        return self.bias
    
    def resetWeights(self):
        self.weights = np.clip(np.random.normal(0, 1/np.sqrt(30), 30).reshape((1,30)), -2, 2)
        self.bias = np.clip(np.random.normal(0, 1/np.sqrt(30), 1).reshape(()), -2, 2)
        
    def eval(self,forewardPass):
        """
        evaluates forewardPass on the training dataset and returns the RMSE (root-mean-square error) 
        """
        results = forewardPass(self.X_test)
        return mean_squared_error(self.y_test, results)
        
        
class Train:
    """
    Evaluates the a neural network for the forward-propagation excersice.
    """
    def __init__(self, train_x, train_y, test_x, test_y, trainFkt, forewardFkt, oneHot, batches):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.trainFkt = trainFkt
        self.forewardFkt = forewardFkt
        self.oneHot = oneHot
        self.batches = batches
        
    def shuffle(self):
        
        
    def get_test(self):
        return self.test_x
    
    def train(self, iterations, eval_):
        
        #plt.figure()
        #plt.title("training")
        train_loss = np.array([])
        test_loss = np.array([])
        for i in range(0,iterations):
            if (i % eval_ == 0 or i == 0):
                loss_test = 0
                loss_train = 0
                for x in range (1,10):
                    ind = np.random.choice(self.test_x.shape[0], 1)
                    y_test = self.test_y[ind]
                    if (self.oneHot):
                        y_test = np.zeros((1, 10))
                        y_test[self.test_y[ind]] = 1
                    res_test = self.forewardFkt(self.test_x[ind]).reshape((1,))
                    loss_test += log_loss(y_test, res_test, labels=[0,1])
                    
                    ind = np.random.choice(self.train_x.shape[0], 1)
                    y_train = self.train_y[ind]
                    if (self.oneHot):
                        y_train = np.zeros((1, 10))
                        y_train[self.train_x[ind]] = 1
                    #import pdb; pdb.set_trace()
                    res_train = self.forewardFkt(self.train_x[ind]).reshape((1,))
                    loss_train += log_loss(y_train, res_train, labels=[0,1])
                train_loss = np.append(train_loss, loss_train / 10.)
                test_loss = np.append(test_loss, loss_test / 10.)
                #print ("train", loss_train / 5.)
                #print ("test", loss_test / 5.)
            ind = np.random.choice(self.train_x.shape[0], 1)
            self.trainFkt(self.train_x[ind], self.train_y[ind])

        #TODO total error
        plt.figure()
        plt.title("Test & Training cross-entropy loss")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        it = np.arange(0., test_loss.shape[0], 1)
        plt.plot(it, train_loss, 'o-', color="r",
             label="Training score")
        plt.plot(it, test_loss, 'o-', color="g",
                 label="Test score")
        plt.legend(loc="best")
class BackwardWeightsMNIST:
    """
    This class provides the weights for the forward-propagation Breast Cancer Wisconsin (Diagnostic) Dataset excersice.
    """
    """
    This class provides the weights for the forward-propagation excersice.
    """
    def __init__(self):
        mnist = fetch_mldata("MNIST original")
        # rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        self.weights0 = np.clip(np.random.normal(0, 1/np.sqrt(30), 39200).reshape((784, 50)), -2, 2)
        self.weights1 = np.clip(np.random.normal(0, 1/np.sqrt(30), 500).reshape((50,10)), -2, 2)
        self.bias0 = np.clip(np.random.normal(0, 1/np.sqrt(30), 50).reshape((50,)), -2, 2)
        self.bias1 = np.clip(np.random.normal(0, 1/np.sqrt(30), 10).reshape((10,)), -2, 2)
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        
    def getTrain(self, trainFkt, forewardFktProba):
        return Train(self.X_train, self.y_train, self.X_test, self.y_test, trainFkt, forewardFktProba, False)
    
    def setWeights(self, i, W):
        if (layer == 0):
            self.weights0 = W
        else:
            self.weights1 = W
    
    def setBias(self, i, b):
        if (layer == 0):
            self.bias0 = b
        else:
            self.bias0 = b
    
    def getWeights(self, layer):
        """
        returns the weights
        """
        if (layer < 0 | layer > 1):
            raise ValueError("there are only 2 layers: 1,0")
        if (layer == 0):
            return self.weights0
        else:
            return self.weights1
    
    def getBias(self, layer):
        """
        returns the bias
        """
        if (layer < 0 | layer > 1):
            raise ValueError("there are only 2 layers: 1,0")
        if (layer == 0):
            return self.bias0
        else:
            return self.bias1
    
    def resetWeights(self):
        self.weights0 = np.clip(np.random.normal(0, 1/np.sqrt(30), 39200).reshape((784, 50)), -2, 2)
        self.weights1 = np.clip(np.random.normal(0, 1/np.sqrt(30), 500).reshape((50,10)), -2, 2)
        self.bias0 = np.clip(np.random.normal(0, 1/np.sqrt(30), 50).reshape((50,)), -2, 2)
        self.bias1 = np.clip(np.random.normal(0, 1/np.sqrt(30), 10).reshape((10,)), -2, 2)
        
    def eval(self,forewardPass):
        """
        evaluates forewardPass on the training dataset and returns the RMSE (root-mean-square error) 
        """
        results = forewardPass(self.X_test)
        return mean_squared_error(self.y_test, results)