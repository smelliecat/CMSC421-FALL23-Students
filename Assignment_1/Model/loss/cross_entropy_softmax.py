import numpy as np

class CrossEntropySoftMaxLoss:
    """
    Implements the combined cross-entropy loss and Softmax activation, commonly used in multi-class classification tasks.

    The cross-entropy loss for multi-class classification with Softmax is defined as: 
    \( - \sum_{c=1}^M y_c \log(p_c) \)
    where \( y_c \) is the true label and \( p_c \) is the predicted probability for each class \( c \).

    Attributes:
        input_dimension (tuple): The shape of the output from the preceding layer in the neural network.
        labels (array-like, optional): Ground truth labels.
        
    Methods:
        set_data(labels): Method to set the labels.
        forward(): Computes the forward pass, calculating the cross-entropy loss.
        backward(): Computes the backward pass, calculating the gradient of the loss with respect to the input.
    """

    def __init__(self, input_dimension, labels=None) -> None:
        self.input_layer = input_dimension
        if labels is not None: # you don't have to pass labels if it is not known at class construction time. (e.g. if you plan to do mini-batches)
            self.set_data(labels)
    
    def set_data(self, labels):
        self.labels = labels
        self.ones_hot = np.zeros((labels.shape[0], labels.max()+1))
        self.ones_hot[np.arange(labels.shape[0]),labels] = 1

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
        # TODO: Compute grad of loss with respect to inputs, and hand this gradient backward to the layer behind. Be careful! Don't exponentiate an arbitrary positive number as it may overflow. 
        input_grad = ...
        self.in_layer.backward(input_grad)



