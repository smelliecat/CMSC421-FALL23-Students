import numpy as np

from Model.layers.linear import LinearLayer

# from activation.relu import ReLU
from activation.sigmoid import Sigmoid
from activation.tanh import Tanh

class OutputLayer(LinearLayer):
    """
    Represents the output layer in a neural network, inheriting from the Linear class.
    It allows the use of activation functions on the output.
    
    Attributes:
        activation: The activation function to be used after the linear operation.
        activated_output: The output after applying the activation function.

    Methods:
        forward(): Performs the forward pass by applying the linear operation and activation function.
        backward(dwnstrm): Performs the backward pass to compute the gradients.
    """

    def __init__(self, input_layer, num_out_features, activation='Linear'):
        super().__init__(input_layer, num_out_features)
        if activation == 'Sigmoid':
            self.activation = Sigmoid(self)
        elif activation == 'Tanh':
            self.activation = Tanh(self)
        # elif activation == 'Linear':
        #     self.activation = 

    def forward(self):
        _out = super().forward()
        # self.activated_output = self.activation.forward()
        return _out

    def backward(self, downstream):
        # activation_grad = self.activation.backward(downstream)
        super().backward(downstream)
