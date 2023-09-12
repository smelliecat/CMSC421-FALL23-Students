import numpy as np
from typing import List
from Model.layers.linear import LinearLayer

class AdamSolver:
    """
    Implements the Adam optimization algorithm.
    Adam Algorithm:
    Adam combines the benefits of both AdaGrad and RMSProp.
    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    where m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t)

    Parameters:
    lr (float): Learning rate
    modules (List[LinearLayer]): List of layers in the model (excluding the input layer)
    beta1 (float, optional): Exponential decay rate for first moment estimate, default is 0.9
    beta2 (float, optional): Exponential decay rate for second moment estimate, default is 0.999
    epsilon (float, optional): Small constant to prevent division by zero, default is 1e-8
    """

    def __init__(self, learning_rate:float, modules: List[LinearLayer], beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8):
        self.learning_rate = learning_rate
        self.modules = modules
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        for module in self.modules:
            module.m = 0
            module.v = 0
       

    def step(self):
        """
        """
        self.t += 1
        for module in self.modules:
            # TODO: Update the first moment (m) and the second moment (v) for each parameter.
            # 'beta1' and 'beta2' are the decay rates for the first and second moments, respectively.
            # Note: 'module.G.mean(axis=0)' computes the mean gradient across the batch.
            module.m = ...
            module.v = ...

            # TODO: Compute bias-corrected first moment (m_hat) and second moment (v_hat)
            # 't' is the timestep which is incremented each time this method is called.
            m_m_hat = ...
            m_v_hat = ...

            # DONE: Update the weights (W) using the bias-corrected moments.
            # The update rule for Adam divides the learning rate-scaled m_hat by the square root of v_hat.
            # We add 'epsilon' to the denominator to prevent division by zero.
            module.W -= self.learning_rate * m_m_hat / (np.sqrt(m_v_hat) + self.epsilon)

            pass
        pass
    pass



 

