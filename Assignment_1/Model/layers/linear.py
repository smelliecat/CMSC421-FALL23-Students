import numpy as np

class LinearLayer:
    """
    A class representing a fully connected (linear) layer in a neural network.
    
    Methods:
        forward(): Computes the forward pass of the layer.
        backward(dwnstrm): Computes the backward pass, propagating the gradient.
    """

    def __init__(self, input_layer, number_out_features) -> None:
        """
        Initialize the layer.

        Parameters:
            input_layer: The preceding layer in the neural network.
            output_dimension: Number of neurons in the current layer.
        
        Raises:
            AssertionError: If input layer dimensions are not a list of 1D linear feature data.
        """
        print("Dim in linear layer", input_layer.output_dimension)
        # print(input_layer.output_dimension)
        assert len(input_layer.output_dimension) == 2, "Input layer must contain a list of 1D linear feature data."
        self.input_layer = input_layer
        num_data, num_in_features = input_layer.output_dimension
        self.output_dimension = np.array([num_data, number_out_features])
        # Declare the weight matrix and initialize it
        self.W = np.random.randn(num_in_features, number_out_features) / np.sqrt(num_in_features)
        print(self.W)


    def forward(self):
        """
        Compute the forward pass for the layer, i.e., compute XW.
        """
        self.input_array = self.input_layer.forward()
        print("Linear Forward", self.input_array)
        self.output_array = self.input_array @ self.W
        return self.output_array

    def backward(self, downstream):
        """
        Compute the backward pass for the layer, propagating the gradient backward.
        """
        # Compute gradient with respect to weights
        self.G = self.input_array[:, :, np.newaxis] * downstream[:, np.newaxis]
        # Compute gradient with respect to inputs
        print("Shape of G:", self.G.shape)
        print("G:", self.G)
        print("W:", self.W)
        # print("Shape of downstream:", downstream.shape)
        # print("Values of downstream:", downstream)
        print("Shape of downstream in LinearLayer:", downstream.shape)
        input_grad = (self.W @ downstream[:, :, np.newaxis]).squeeze(axis=-1)
        print('Shape of input_grad in LinearLayer: ', input_grad.shape)
        print('Shape of self.input_array in LinearLayer:', self.input_array.shape)
        self.input_layer.backward(input_grad)
