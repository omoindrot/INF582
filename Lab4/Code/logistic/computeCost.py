from numpy import *
from sigmoid import sigmoid


def computeCost(theta, X, y):
    # Computes the cost using theta as the parameter
    #  for logistic regression.

    m = X.shape[0] # number of training examples

    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment 
    #				for more details).

    temp = X.dot(theta)
    temp = sigmoid(temp)

    J += y.dot(log(temp)) + (1-y).dot(log(1-temp))

    J = -J/m

    return J
