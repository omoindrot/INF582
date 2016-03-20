from numpy import *
from euclideanDistance import euclideanDistance


def find_closest_neighbours(data, N):
    
    closest_neighbours = zeros((data.shape[0], N))

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Find the N closest instances of each instance
    #               using the euclidean distance.
    
    for i in range(data.shape[0]):
        distances = zeros(data.shape[0])
        for j in range(data.shape[0]):
            distances[j] = euclideanDistance(data[i], data[j])

        closest_neighbours[i] = distances.argsort()[1:N+1]

    # =============================================================
    
    return closest_neighbours
