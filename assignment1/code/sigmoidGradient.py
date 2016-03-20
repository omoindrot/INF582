from numpy import *
from sigmoid import sigmoid


def sigmoidGradient(z):

    # sigmoidGradient returns the gradient of the sigmoid function evaluated at z

    g = zeros(z.shape)
    # =========================== DONE ==================================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z.
    g += sigmoid(z) * (1 - sigmoid(z))
    
    return g


def reluGradient(z):

    # reluGradient returns the gradient of the relu function evaluated at z
    return 1.0*(z > 0)
