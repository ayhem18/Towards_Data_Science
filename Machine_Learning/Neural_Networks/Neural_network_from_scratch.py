import numpy as np
import pandas as pd

default_hidden_dimensions = np.array([4, 4])

class NeuralNetwork:
    def __init__(self, hidden_layers: int=2, hidden_dimensions: np.array=None, is_regularized: bool=False, reg_par: int=0):
        # if the number of hidden layers is passed, then the dimension must be explicitly indicated
        # and size of the hidden_dimensions array is
        if hidden_layers != None or hidden_dimensions != None:
            assert hidden_layers != None and hidden_dimensions != None and len(hidden_dimensions) == hidden_layers
        
        # the regularization parameters must be positive
        assert (not is_regularized and reg_par == 0) or is_regularized and reg_par > 0

        # set the default hidden_dimensions
        if hidden_dimensions is None: 
            self.hidden_dimensions = default_hidden_dimensions
        
        else: 
            self.hidden_dimensions = hidden_dimensions
        
        self.hidden_layers = hidden_layers
        self.is_regularized = is_regularized
        self.reg_par = reg_par
        
        # create fields for later use
        
        # field to save the total number of layers
        L = hidden_layers + 2
        # field to save the number of output layer's units
        K = None
        # field to save the dimensions of all layers including input and output layers
        self.dimensions = None
        # field to save the weights
        self.thetas = None
        # field to save the number of features per training sample
        self.n_feature = None

    
    def initialize_thetas(self):
        


    def fit(self, X: np.array, y: np.array, num_class):
        
        assert num_class >= 2
        
        if num_class == 2: 
            # binary classification
            K = 1
        else:
            # multi-class classification
            K = num_class

        self.n_feature = X.shape[1]
        self.thetas = 

        
        


        

        