"""
define modules of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) for MNIST classification.
    
    Attributes:
        conv: Convolutional layers.
        fc: Fully connected layer for classification.
    """

    def __init__(self, args):
        super(CNNModel, self).__init__()
        """
        Initialize the CNN model with given arguments.
        
        Args:
            args: Arguments containing hyperparameters.
        """
        
        # Define the model architecture here
        # MNIST image input size batch * 28 * 28 (one input channel)
        
        # TODO Define CNN layers below
        self.conv = nn.Sequential(
           ...
        )

        # TODO Define fully connected layer below
        input_size = ...
        output_size = ...  # Example size, adjust as needed
        self.fc = nn.Linear(input_size, output_size)

    # Feed features to the model
    def forward(self, x):  # default
        """
        Forward pass of the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            result: Output tensor of shape (batch_size, num_classes)
        """
        # TODO Feed input features to the CNN models defined above
        x_out = ...
        
        # TODO Flatten tensor code
        x = ...

        # Fully connected layer (Linear layer)
        result = self.fc(x)  # predict y
        
        return result
