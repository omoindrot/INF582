from numpy import *


def addColumnOne(X):
    # addColumnOne adds a column of ones at the end of the matrix X
    (n, p) = X.shape
    res = zeros((n, p+1))

    res[:, :p] += X
    res[:, p] += 1.0

    return res


def removeLastColumn(X):
    # removeColumnOne removes a column of ones at the end of the matrix X
    (n, p) = X.shape
    res = zeros((n, p-1))

    res += X[:, :p-1]

    return res
