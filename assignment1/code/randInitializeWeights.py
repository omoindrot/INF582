from numpy import *


def randInitializeWeights(layers):

    num_of_layers = len(layers)
    # epsilon = 0.12
        
    Theta = []
    for i in range(num_of_layers-1):
        # ====================== DONE ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #

        # We initialize the transpose of W
        W_T = zeros((layers[i] + 1, layers[i+1]), dtype='float32')

        # DONE: change epsilon here to a more appropriate value (Gorot initialization)
        epsilon = 4*sqrt(6)/sqrt(layers[i+1]+layers[i])

        # The first row of W_T (i.e the first column of W) should be initialized to 0 (weights of the bias)
        # The rest is initialized uniformly in [-epsilon, +epsilon] with Gorot initialization
        W_T[1:] += 2*epsilon*random.rand(layers[i], layers[i+1])-epsilon
        W = W_T.T
        Theta.append(W)
                
    return Theta
