from numpy import *

def frobenius(Y,X): # X is the original and Y the reduced
    #returns the percentage of the frobenius norm compared to the original
    fro = linalg.norm(X-Y,ord='fro')/linalg.norm(X, ord='fro')
    return fro
