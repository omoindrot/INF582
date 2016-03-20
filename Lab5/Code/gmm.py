# -*- coding: utf-8 -*-
"""

@author: jb
"""
from gaussian import gaussian


def gmm(alpha, lambd, sigma, x):
    res = 0
    for i in range(alpha.shape[1]):
        res += alpha[i]*gaussian(x, sigma[i], lambd[i])

    return res