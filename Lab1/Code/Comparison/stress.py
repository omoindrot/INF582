from numpy import *

def stress(Y,X): # X is the original and Y the reduced
    Sim = dot(X,transpose(X)) # similarity matrix 
    A = diag(Sim)
    A = A.reshape(len(A),1)
    # the distance matrix 
    distX = sqrt(tile(transpose(A),(X.shape[0],1))+tile(A,(1,X.shape[0])) - 2 * Sim)
    Sim = dot(Y,transpose(Y))   
    A = diag(Sim)
    A = A.reshape(len(A),1)
    distY = real(sqrt(tile(transpose(A),(X.shape[0],1))+tile(A,(1,X.shape[0])) - 2 * Sim))
    stress = sqrt(sum(sum(power((distY-distX),2*ones(distY.shape)))) / sum(sum(power(distX,2*ones(distX.shape)))))   
    return stress
    
