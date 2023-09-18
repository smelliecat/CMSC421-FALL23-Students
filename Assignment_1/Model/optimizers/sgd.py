import numpy as np
from typing import List
from Model.layers.linear import LinearLayer
class SGDSolver:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
    
    Algorithm Description:
    SGD updates each parameter (W) based on the gradient (G) of the objective function 
    with respect to that parameter. The formula for the parameter update is:
    
    W = W - lr * mean(G)
    
    where lr is the learning rate and mean(G) is the mean of the gradients.
    
    Parameters:
    - learning_rate (float): Learning rate.
    - modules (List): List of layers in the model, excluding the input layer. 
                       All layers should have a parent class of LinearLayer.
    """
    def __init__(self, learning_rate:float, modules:List[LinearLayer]):
        # TODO: Initialize the learning_rate and modules attributes. Replace 'None' with appropriate code.
        self.learning_rate = learning_rate
        self.modules = modules

    def step(self):
        """
        Perform a single optimization step, updating the parameters of all layers in 'modules'.
        """

        """
        Perform a single optimization step.
        
        Description:
        ------------
        The method should update the parameters of all layers in the 'modules' list according
        to the SGD update rule.
        """
        

        for module in self.modules:
            # TODO: Loop through each module in self.modules and update its weight `W` using its gradient `G`.
            module.W = ...
            pass

