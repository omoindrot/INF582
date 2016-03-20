import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y)  # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape  # dimensions of the dataset
    totalMean = np.mean(X, 0).reshape((dim, 1))  # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudo-code on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.

    # partition: liste de tableau ou chaque tableau contient les indices des donnees d'une meme classe
    partition = [np.where(Y == i) for i in classLabels]

    # barycentres de chaque classe
    classMean = [(np.mean(X[idx], axis=0), np.size(idx)) for idx in partition]

    # calcul de Sw
    Sw= np.zeros((dim, dim))
    for idx in partition:
        Sw += np.cov(X[idx], rowvar=0)*np.size(idx)

    # calcul de Sb
    Sb = np.zeros((dim, dim))
    for mu, classSize in classMean:
        mu = mu.reshape((dim, 1))
        Sb += classSize*(mu - totalMean).dot((mu - totalMean).T)

    # Calcul des valeurs propres et vecteurs propres
    S = np.dot(linalg.inv(Sw), Sb)
    eigval, eigvec = linalg.eig(S)

    idx = eigval.argsort()[::-1]
    print np.real(eigval[idx])
    eigvec = eigvec[:, idx]
    W = np.real(eigvec[:, :classNum-1])
    print W.shape

    projected_centroid = [mu.dot(W) for mu, classSize in classMean]
    X_lda = X.dot(W)

    # Plotting
    fig = plt.figure(1)
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=Y)
    plt.show()
    fig = plt.figure(2)
    plt.scatter(X[:, 5], X[:, 7], c=Y)
    plt.show()

    return W, projected_centroid, X_lda
    
    
    
    
    
    
    # =============================================================

    return W, projected_centroid, X_lda