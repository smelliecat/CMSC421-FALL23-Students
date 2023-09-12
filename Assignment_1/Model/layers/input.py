import numpy as np
# -------------------------------
# DO NOT MODIFY THIS FILE!!!!!  |
# -------------------------------
class InputLayer:
    """
    Represents the input layer of a neural network. The input layer essentially acts as
    a pass-through for the input data and does not perform any transformations.

    Attributes:
        W: Identity matrix corresponding to the dimensions of the input data. 
           This is included for compatibility with other layers but is not used.
        output_dimension: The dimensions of the output (same as input in the case of InputLayer).

    Methods:
        forward(input_data): Passes the input data through the layer without altering it.
        backward(downstream): Placeholder function; no actual backward pass computations 
                              are performed in the input layer.
    """
     
    def __init__(self, data_layer) -> None:
        """
        Initialize the InputLayer with data dimensions.

        Parameters:
        -----------
            data_layer: The layer that feeds data into this InputLayer.
        """
        # -------------------------------
        # DO NOT MODIFY THIS FILE!!!!!  |
        # -------------------------------
        num_data, num_in_features = (data_layer.output_dimension)
        self.output_dimension = np.array([num_data, num_in_features])
        self.data_layer = data_layer
    
    def forward(self):
        """
        Performs the forward pass by simply passing the input data through the layer without altering it.
        
        Returns:
        --------
            numpy array: The input data passed through.
        """
        # -------------------------------
        # DO NOT MODIFY THIS FILE!!!!!  |
        # -------------------------------
        self.input_array = self.data_layer.forward()
        print(self.output_dimension)
        return self.input_array

    def backward(self, downstream):
        # -------------------------------
        # DO NOT MODIFY THIS FILE!!!!!  |
        # -------------------------------
        """
        Placeholder function for backward pass.
        
        As this is an InputLayer, no actual backward computations are performed.

        Parameters:
        -----------
            downstream: The gradient received from the downstream layer, ignored in InputLayer.
        """
        pass


