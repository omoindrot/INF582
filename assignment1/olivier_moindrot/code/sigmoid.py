from numpy import *


def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = zeros(shape(z))

    # ============================= DONE ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.

    g += 1.0 / (1.0 + exp(-z))

    return g


def relu(z):

    # RELU returns rectified linear unit evaluated at z
    return maximum(z, 0)
