import numpy as np
from math import fabs, fsum
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

class ForewardWeightsBCW:
    """
    This class provides the weights for the forward-propagation Breast Cancer Wisconsin (Diagnostic) Dataset excersice.
    """
    """
    This class provides the weights for the forward-propagation excersice.
    """
    def __init__(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        self.weights = log_reg.coef_
        self.bias = log_reg.intercept_[0]
        self.X_test = X_test
        self.y_test = y_test
    
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
        
    def eval(self,forewardPass):
        """
        evaluates forewardPass on the training dataset and returns the RMSE (root-mean-square error) 
        """
        results = forewardPass(self.X_test)
        return mean_squared_error(self.y_test, results)
        

class ForewardWeightsMNIST:
    """
    This class provides the weights for the forward-propagation excersice.
    """
    def __init__(self):
        mnist = fetch_mldata("MNIST original")
        # rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                            solver='sgd', tol=1e-4, random_state=1,
                            learning_rate_init=.1)

        mlp.fit(X_train, y_train)
        self.mlp=mlp
        self.weights = mlp.coefs_
        self.biases = mlp.intercepts_
        self.X_test = X_test
        self.y_test = y_test
    
    def getWeights(self, layer):
        """
        returns the weights for the layer
        """
        if (layer < 0 | layer > 1):
            raise ValueError("there are only 2 layers: 1,0")
        return self.weights[layer]
    
    def getBias(self, layer):
        """
        returns the biases for the layer
        """
        if (layer < 0 | layer > 1):
            raise ValueError("there are only 2 layers: 1,0")
        return self.biases[layer]
    
    def getRandom(self):
        idx = np.random.randint(self.y_test.shape[0], size=2)
        r_x = self.X_test[idx,:]
        r_y = self.y_test[idx]
        return (r_x, r_y)
    
    def visualize(self,forewardPassPropa):
        (r_x,r_y) = self.getRandom()
        pred0 = forewardPassPropa(r_x[0])

        plt.figure(figsize=(20,4))
        plt.subplot(1, 5, 1)
        plt.imshow(np.reshape(r_x[0], (28,28)), cmap=plt.cm.gray)
        plt.title('Label: %i\n' % r_y[0], fontsize = 20)
        
        plt.subplot(1, 5, 2)
        plt.bar(np.arange(0,10,1), pred0, align='center', alpha=0.5)
        plt.ylim(0, 1)
        plt.title('Prediction\n', fontsize = 20)
        
        pred1 = forewardPassPropa(r_x[0])
        plt.subplot(1, 5, 3)
        plt.imshow(np.reshape(r_x[1], (28,28)), cmap=plt.cm.gray)
        plt.title('Label: %i\n' % r_y[1], fontsize = 20)
        
        plt.subplot(1, 5, 4)
        plt.bar(np.arange(0,10,1), pred1)
        plt.ylim(0, 1)
        plt.title('Prediction\n', fontsize = 20)
    
    def eval(self,forewardPassClass):
        """
        evaluates forewardPass on the training dataset and returns the RMSE (root-mean-square error) 
        """
        print("mse real:",mean_squared_error(self.y_test, self.mlp.predict(self.X_test)))
        results = forewardPassClass(self.X_test)
        return mean_squared_error(self.y_test, results)