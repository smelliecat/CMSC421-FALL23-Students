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
from Data.generator import q1_a
from Model.evaluate.evaluate import evaluate_model


Number_of_iterations = 300 # Experiment to pick your own number of ITERATIONS
Step_size = 0.000001 # Experiment to pick your own STEP number

class Network(BaseNetwork):
    def __init__(self, data_layer):
        super().__init__()
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, 1)
        self.bias_layer1 = BiasLayer(self.hidden_layer1)
        self.output_layer1 = OutputLayer(self.bias_layer1, 1)
        self.set_output_layer(self.output_layer1)


# To get you started we built the network for you!! Please use the template file to finish answering the question
