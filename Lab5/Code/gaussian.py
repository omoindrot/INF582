# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 18:33:29 2014

@author: jb
renvoie la valeur de la distribution en une valeur donnée
"""
from numpy import *
from math import pi


def loggaussian(x, var, moyenne):
    # x: vecteur colonne
    # if linalg.det(sigma)>0:
    re = 0
    for i in range(shape(var)[0]):
        if var[i]>0:
            re = re - 0.5*((x[i]-moyenne[i]))**2/(var[i]) - 0.5*log(var[i]) - 0.5*log((2*pi)**shape(x)[0])
        else:
            if moyenne[i]==x[i]:
                return 0
            else:
                return 999999
    return -re   
   

def loggmm(x,var,moyenne,alpha,K):
    res2 = zeros((K))
    for i in range(K):
        res2[i]=loggaussian(x,var[i,:],moyenne[i,:]) - log(alpha[i])
    return min(res2)     #HERE there was a mistake
    
    