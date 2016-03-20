from numpy import *
from sigmoid import sigmoid
from addColumnOne import addColumnOne


def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ DONE ================================
    # You need to return the following variables correctly
    p = zeros((1, m))

    h = X
    for i in range(num_layers-1):
        h = sigmoid(addColumnOne(h).dot(Theta[i].T))

    p = argmax(h, axis=1)
    return p
