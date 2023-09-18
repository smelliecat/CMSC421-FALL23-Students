import numpy as np

class Tanh:
    """
    Implements the hyperbolic tangent (tanh) activation function.
    
    The tanh function is defined as: f(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    
    Attributes:
        in_layer: The layer that provides the input to this activation function.
        
    Methods:
        forward(): Applies the tanh activation function to the output of the input layer.
        backward(downstream): Computes the gradient of the loss with respect to the input, 
                           which is then passed back to the previous layers.
    """

    @staticmethod
    def forward(input_array):
        # Apply the tahn activation function to the output of the input layer. 
        # You can NOT use the np.tahn function.
        output_array = ...
        return output_array
    
    @staticmethod
    def backward(downstream, input_array=None):
        # Compute the gradient of the loss with respect to the input
        tanh_grad = ...
        input_grad = downstream * tanh_grad
        return input_grad


# They can NOT use the np.tahn function

