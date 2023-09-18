import numpy as np

from Model.layers.linear import LinearLayer

from activation.relu import ReLU
from activation.sigmoid import Sigmoid
from activation.tanh import Tanh

class HiddenLayer(LinearLayer):
    """
    Represents a hidden layer in a neural network. Inherits from the LinearLayer class
    and adds activation functionality.
    
    Attributes:
    -----------
    activation: An activation function object. This can be ReLU, Sigmoid, or Tanh.
    
    Methods:
    --------
    forward(): Performs the forward pass, including linear transformation and activation.
    backward(downstream): Performs the backward pass, including both activation and linear gradients.
    """
    def __init__(self, input_dimension, output_dimension) -> None:
        """
        Initializes the HiddenLayer.
        
        Parameters:
        -----------
        input_dimension: The number of input features.
        output_dimension: The number of output features.
        activation (optional): The activation function to use ('ReLU', 'Sigmoid', or 'Tanh'). Default is 'ReLU'.
        """
        super().__init__(input_dimension, output_dimension)

    def forward(self):
        """
        Performs the forward pass by first conducting the linear transformation and then the activation.
        
        Returns:
        --------
        _out: The linearly transformed data.
        """
        _out = super().forward()
        return _out
    
    def backward(self, downstream):
        """
        Performs the backward pass by propagating the gradient through the activation function
        and then through the linear transformation.
        
        Parameters:
        -----------
        downstream: The gradient of the loss with respect to the output of this layer.
        """
        # TODO: Implement the backward pass.
        super().backward(downstream=downstream)
        