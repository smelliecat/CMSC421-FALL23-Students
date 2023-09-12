import numpy as np

class CrossEntropyLoss:
    """
    Implements the cross-entropy loss function, commonly used in classification tasks.

    The cross-entropy loss for binary classification is defined as: 
    \( - (y \log(p) + (1 - y) \log(1 - p)) \)

    Attributes:
        input_layer: The preceding layer of the neural network.
        labels: Ground truth labels.
        
    Methods:
        set_data(labels): Method to set the labels.
        forward(): Computes the forward pass, calculating the cross-entropy loss.
        backward(): Computes the backward pass, calculating the gradient of the loss.
    """

    def __init__(self, input_dimension, labels=None) -> None:
        self.input_layer = input_dimension
        self.labels = labels
    
    def set_data(self, labels):
        self.labels = labels

    def forward(self):
        # TODO: Implement the forward pass to compute the loss.
        self.in_array = self.input_layer.forward()
        self.num_data = self.in_array.shape[1]
        # TODO: Compute the result of mean squared error, and store it as self.out_array
        self.out_array = ...
        return self.out_array


    def backward(self):
        """
        """
        # TODO: Compute grad of loss with respect to inputs, and hand this gradient backward to the layer behind
        input_grad = ... 
        self.input_layer.backward(input_grad)
        pass



