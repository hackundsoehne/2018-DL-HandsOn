import numpy as np
from math import fabs, fsum
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import scale
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
        scaled = scale(cancer.data)
        X_train, X_test, y_train, y_test = train_test_split(scaled, cancer.target, stratify=cancer.target, random_state=42)
        self.log_reg = LogisticRegression()
        self.log_reg.fit(X_train, y_train)
        self.weights = np.clip(np.random.normal(0, 1/np.sqrt(30), 30).reshape((1,30)), -2, 2)
        self.bias = np.clip(np.random.normal(0, 1/np.sqrt(30), 1).reshape(()), -2, 2)
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        
    def getTrainID(self, trainFkt, forewardFktProba, batch):
        X_ID_test = np.ones(self.X_test.shape)
        X_ID_test[:,0] = self.y_test
        X_ID_train = np.ones(self.X_train.shape)
        X_ID_train[:,0] = self.y_train
        return Train(X_ID_train, self.y_train.reshape(-1,1), X_ID_test, self.y_test.reshape(-1,1), trainFkt, forewardFktProba, False, batch)
        
    def getTrain(self, trainFkt, forewardFktProba):
        return Train(self.X_train, self.y_train.reshape(-1,1), self.X_test, self.y_test.reshape(-1,1), trainFkt, forewardFktProba, False, 20)
    
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
        
    def solveWeights(self):
        self.weights = self.log_reg.coef_
        self.bias = self.log_reg.intercept_[0]
        
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
    def __init__(self, train_x, train_y, test_x, test_y, trainFkt, forewardFkt, oneHot, batchsize):
        (sh_train_x, sh_train_y) = self.shuffle(train_x, train_y)
        self.train_x = sh_train_x
        self.train_y = sh_train_y
        (sh_test_x, sh_test_y) = self.shuffle(test_x, test_y)
        self.test_x = sh_test_x
        self.test_y = sh_test_y
        self.trainFkt = trainFkt
        self.forewardFkt = forewardFkt
        self.oneHot = oneHot
        self.batchsize = batchsize
        self.train_id = 0
        self.test_id = 0
        
    def shuffle(self, X, Y):
        #idx = np.random.permutation(X.shape[0])
        #x,y = X[idx], Y[idx]
        #return (x,y)
        return (X, Y)
        
    def getNextTrain(self):
        if (self.train_id + self.batchsize > self.train_x.shape[0]):
            (sh_x, sh_y) = self.shuffle(self.train_x, self.train_y)
            self.train_x = sh_x
            self.train_y = sh_y
            self.train_id = 0
        next_x = self.train_x[self.train_id:self.train_id+self.batchsize]
        next_y = self.train_y[self.train_id:self.train_id+self.batchsize]
        self.train_id = self.train_id+self.batchsize
        return (next_x, next_y)
    
    def getNextTest(self):
        if (self.test_id + self.batchsize > self.test_x.shape[0]):
            (sh_x, sh_y) = self.shuffle(self.test_x, self.test_y)
            self.test_x = sh_x
            self.test_y = sh_y
            self.test_id = 0
        next_x = self.train_x[self.test_id:self.test_id+self.batchsize]
        next_y = self.train_y[self.test_id:self.test_id+self.batchsize]
        self.test_id = self.test_id+self.batchsize
        return (next_x, next_y)
    
    def train(self, iterations, eval_):
        
        #plt.figure()
        #plt.title("training")
        train_loss = np.array([])
        test_loss = np.array([])
        for i in range(0,iterations):
            if (i % eval_ == 0 or i == 0):
                (test_x, test_y) = self.getNextTest()
                (train_x, train_y) = self.getNextTrain()
                if (self.oneHot):
                    #convert to one-hot
                    nb_classes = 10
                    #import ipdb; ipdb.set_trace()
                    test_y = np.eye(nb_classes)[test_y.astype(int)]
                    train_y = np.eye(nb_classes)[train_y.astype(int)]
                res_test = self.forewardFkt(test_x)
                res_train = self.forewardFkt(train_x)
                #loss_test = np.mean(log_loss(test_y, res_test, labels=[0,1]))
                #import ipdb; ipdb.set_trace()
                loss_test = np.mean((0.5)*np.power(test_y - res_test, 2))
                loss_train = np.mean((0.5)*np.power(train_y - res_train, 2))
                #import ipdb; ipdb.set_trace()
                train_loss = np.append(train_loss, loss_train)
                test_loss = np.append(test_loss,loss_test)
                #print ("train", loss_train / 5.)
                #print ("test", loss_test / 5.)
            (train_x, train_y) = self.getNextTrain()
            if (self.oneHot):
                    #convert to one-hot
                    nb_classes = 10
                    train_y = np.eye(nb_classes)[train_y.astype(int)]
            self.trainFkt(train_x, train_y)

        #TODO total error
        plt.figure()
        plt.title("Test & Training MSE")
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
        return Train(self.X_train, self.y_train, self.X_test, self.y_test, trainFkt, forewardFktProba, True, 64)
    
    def setWeights(self, layer, W):
        if (layer == 0):
            self.weights0 = W
        else:
            self.weights1 = W
    
    def setBias(self, layer, b):
        if (layer == 0):
            self.bias0 = b
        else:
            self.bias1 = b
    
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