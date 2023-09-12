from Model.layers.network import BaseNetwork
import numpy as np
import matplotlib.pyplot as plt
from Model.layers.input import InputLayer
from Model.layers.hidden import HiddenLayer
from Model.loss.square_loss import SquareLoss
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q1
from Model.evaluate.evaluate import evaluate_model

class Network(BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        data = data_layer.forward()
        self.input_layer = SOME_LAYER
        self.hidden = SOME_LAYER
        self.bias = SOME_LAYER
        self.output_layer1 = SOME_LAYER
        
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.MY_MODULE_LIST = layers.ModuleList()
        # for i in range(N):
        #     self.MY_MODULE_LIST.append(layers.Linear(...))
        
        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.MY_LAST_LAYER)

class Trainer:
    def __init__(self):
        pass
    
    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers. 
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        #TODO: construct your network here
        network = Network(...)
        return network
    
    def net_setup(self, training_data):
        x, y = training_data
        #TODO: define input data layer
        self.data_layer = ...
        #TODO: construct the network. you don't have to use define_network.
        self.network = ...
        #TODO: use the appropriate loss function here
        self.loss_layer = ...
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optimizer = ...
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        loss = ...
        return loss
    def get_num_iters_on_public_test(self):
        #TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 30000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.

        # you have to return train_losses for the function
        return train_losses
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        pass
    else:
        #DO NOT CHANGE THIS BRANCH! 
        pass

if __name__ == "__main__":
    main()
    pass
