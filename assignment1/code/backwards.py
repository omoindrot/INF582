from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from addColumnOne import addColumnOne, removeFirstColumn
from roll_params import roll_params
from unroll_params import unroll_params


def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)
  
    # You need to return the following variables correctly 
    Theta_grad = [zeros(w.shape) for w in Theta]

    # ================================ DONE ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((m, num_labels))
    for i in range(m):
        yv[i][y[i]] += 1

    # ================================ DONE ================================
    # In this point implement the backpropagation algorithm

    # In this point calculate the cost of the neural network (feedforward)

    # Step 1: Initialization of useful variables

    # Z and A will store the hidden states of the network, as lists of matrices, of size num_layers
    A = [addColumnOne(X)]
    Z = [addColumnOne(X)]

    # delta will store the delta for each layer from the last to the second layer (in reverse order)
    delta = []

    # Step 2: Feedforward
    for i in range(num_layers-1):
        h = A[i].dot(Theta[i].T)
        Z.append(h)
        h = addColumnOne(sigmoid(h))
        A.append(h)


    # Step 3: Backpropagation
    d = removeFirstColumn(A[-1]) - yv
    delta.append(d)

    for i in range(num_layers-2, 0, -1):
        d = removeFirstColumn(d.dot(Theta[i])) * sigmoidGradient(Z[i])
        delta.append(d)

    delta.reverse()
    # delta is of size num_layers-1 (no delta for the input layer)

    for i in range(num_layers-1):
        Theta_grad[i] += delta[i].T.dot(A[i])
        # DONE: no regularization on the bias weights !!
        Theta_grad[i] += lambd * Theta[i]
        for j in range(Theta[i].shape[0]):
            Theta_grad[i][j, 0] -= lambd * Theta[i][j, 0]
        Theta_grad[i] /= m

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad
