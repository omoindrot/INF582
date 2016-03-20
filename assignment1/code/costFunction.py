import numpy as np
from sigmoid import sigmoid
from addColumnOne import addColumnOne, removeFirstColumn
from roll_params import roll_params


def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    # You need to return the following variables correctly 
    J = 0
    
    # ================================ DONE ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = np.zeros((m, num_labels))
    for i in range(m):
        yv[i][y[i]] += 1
  

    # ================================ DONE ================================
    # In this point calculate the cost of the neural network (feedforward)

    # Step 1: Initialization of useful variables

    # H will store the hidden states of the network, H is a list of matrices, of size num_layers
    H = [X]

    # Step 2: Feedforward

    for i in range(num_layers-1):
        h = sigmoid(addColumnOne(H[i]).dot(Theta[i].T))
        H.append(h)

    # The end layer is H[num_layers]
    yv_pred = H[num_layers-1]

    # Step 3: Compute cost
    # We create the variable S, a matrix of size (m, K) which we will sum afterwards
    S = np.zeros((m, num_labels))
    temp = np.log(yv_pred)
    temp = yv*temp

    temp2 = np.log(1.0-yv_pred)
    temp2 = (1.0-yv)*temp2

    S += - temp - temp2

    J += np.sum(S)
    J = J/m

    reg = 0
    for i in range(num_layers-1):
        # No regularization on the bias weights
        reg += np.sum(removeFirstColumn(Theta[i])**2)

    J += lambd * reg / (2.0 * m)
    return J

    

