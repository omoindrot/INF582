from numpy import *

def variance(Y,X): # X is the original and Y the reduced
    # usualy we compare the ratio of the sum of first k eigenvectors to 
    # all the eigenvectors but since this is going to be used 
    # for a comparison of non-pca appraches we are going to compare all the
    # eigen vectors between the compressed and the original 

    Xcentered = X - tile(mean(X,0),(X.shape[0],1)) #center data
    Cov = dot(transpose(Xcentered),Xcentered) # covariance matrix
    eigOrig= linalg.eig(Cov)[0]
    Ycentered = Y - tile(mean(Y,0),(Y.shape[0],1)) #center data
    Cov = dot(transpose(Ycentered),Ycentered) # covariance matrix
    eigComp = linalg.eig(Cov)[0]
    var = sum(eigComp) / sum(eigOrig)
    return var
