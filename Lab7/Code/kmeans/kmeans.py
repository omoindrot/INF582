from numpy import *
from euclideanDistance import euclideanDistance
from simpleInitialization import simpleInitialization


def kmeans(X, k):
    # Intialize centroids
    centroids = simpleInitialization(X, k)
    
    # Initialize variables
    iterations = 0
    oldCentroids = None
    labels = zeros(X.shape[0])
    
    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Run the main k-means algorithm. Follow the steps 
    #               given in the description. Compute the distance 
    #               between each instance and each centroid. Assign 
    #               the instance to the cluster described by the closest
    #               centroid. Repeat the above steps until the centroids
    #               stop moving or reached a certain number of iterations
    #               (e.g., 100).

    notConverged = True
    while notConverged:
        # update labels
        old_labels = labels.copy()
        for i in range(X.shape[0]):
            dist = inf
            for j in range(k):
                d = euclideanDistance(X[i], centroids[j])
                if d < dist:
                    labels[i] = j
                    dist = d

        # update centroids
        centroids = zeros((k, X.shape[1]))
        num_centroids = zeros((k, 1)) + 1e-8
        for i in range(X.shape[0]):
            centroids[labels[i]] += X[i]
            num_centroids[labels[i]] += 1.
        centroids = centroids/num_centroids

        iterations += 1
        if (old_labels == labels).all() or iterations>500:
            notConverged = False
            print iterations

    
    # ===============================================================
        
    return labels


def SSE(X, k, labels):
    centroids = zeros((k, X.shape[1]))
    num_centroids = zeros((k, 1)) + 1e-8
    for i in range(X.shape[0]):
        centroids[labels[i]] += X[i]
        num_centroids[labels[i]] += 1.
    centroids = centroids / num_centroids

    res = 0.
    for i in range(X.shape[0]):
        res += euclideanDistance(X[i], centroids[labels[i]])
    res /= X.shape[0]

    return res
