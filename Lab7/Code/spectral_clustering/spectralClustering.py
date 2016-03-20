from numpy import *
from scipy.cluster.vq import kmeans
from euclideanDistance import euclideanDistance


def spectral_clustering(W, k):

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Perform spectral clustering to partition the
    #               data into k clusters. Implement the steps that
    #               are described in Algorithm 2 on the assignment.

    L = diag(sum(W, axis=0)) - W
    w, v = linalg.eig(L)

    y = real(v[:, w.argsort()[:k]])

    clusters, _ = kmeans(y, k)

    labels = zeros(y.shape[0])
    for i in range(y.shape[0]):
        dist = inf
        for j in range(k):
            distance = euclideanDistance(y[i], clusters[j])
            if distance < dist:
                dist = distance
                labels[i] = j
    # =============================================================

    return labels
