from gaussian import loggmm, loggaussian
from numpy import *


def classify_gmm(features, var_mix, mean_mix, alpha_mix, K):
    # features is a vector of features, shape (p,)
    p = features.shape[0]

    # var_mix is a matrix of shape (10, K_max, p)
    # mean_mix is a matrix of shape (10, K_max, p)
    # alpha_mix is a matrix of shape (10, K_max)

    res = zeros(10)
    for i in range(10):
        res[i] = loggmm(features, var_mix[i], mean_mix[i], alpha_mix[i], K)

    return argmax(-res)
