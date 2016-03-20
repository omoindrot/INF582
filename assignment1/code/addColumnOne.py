from numpy import *


def addColumnOne(X):
    # addColumnOne adds a column of ones at the end of the matrix X
    (n, p) = X.shape
    res = zeros((n, p+1))

    res[:, 1:] += X
    res[:, 0] += 1.0

    return res


def removeFirstColumn(X):
    # removeFirstColumn removes a column of ones at the beginning of the matrix X
    (n, p) = X.shape
    res = zeros((n, p-1))

    res += X[:, 1:]

    return res
