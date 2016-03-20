# -*- coding: utf-8 -*-
"""

@author: jb
"""
from numpy import *

from gaussian import loggaussian, loggmm

# this function performs EM


def mix_gmm_em(data, K):
    m, p = data.shape

    W = zeros((K, m), dtype=float64)
    mu_K = zeros((K, p), dtype=float64)
    var_K = zeros((K, p), dtype=float64)
    
    # initialisation by bagging the dataset in K bins
    alpha = ones((K,))
    alpha /= K
    for i in range(K):
        mu_K[i, :] = sum(data[i*m/K:(i+1)*m/K, :], axis=0)
        mu_K[i, :] = mu_K[i, :]*K/m
        Mat = data[i*m/K:(i+1)*m/K, :] - mu_K[i, :]
        var_K[i, :] = diag(dot(Mat.T, Mat)*K/m)

    # epsilon: nombre d'échantillons qui ont migré entre deux itérations
    epsilon = 2
    num_iterations = 0

    while epsilon > 1:

        W_old = W
        W = zeros((K, m), dtype=float64)

        # expectation
        for i in range(m):
            temp = zeros(K)
            for j in range(K):
                temp[j] = loggaussian(data[i], var_K[j], mu_K[j])
            W[argmin(temp), i] = 1

        epsilon = trace(dot((W-W_old).T, W-W_old))

        # maximization
        idx = [where(W[i, :] == 1) for i in range(K)]

        for j in range(K):
            mu_K[j] = mean(data[idx[j]], axis=0)
            var_K[j] = var(data[idx[j]], axis=0)

            # update alpha
            alpha[j] = float(size(idx[j]))/float(m)

        num_iterations += 1

    # print "num_iterations", num_iterations

    # to be completed
        
    return mu_K, var_K, alpha
