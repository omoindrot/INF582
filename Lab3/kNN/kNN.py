from numpy import *
from euclideanDistance import euclideanDistance


def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. Calculate the distance betweet y and each 
    #               row of X, find  the k closest observations and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance between two vectors A and B use
    #       use the euclideanDistance(A,B) function.
    #

    (n, p) = X.shape

    distances = zeros(n)
    for i in range(n):
        distances[i] = euclideanDistance(X[i], y)

    indexes = distances.argsort()
    top_indexes = indexes[:k]

    knn_labels = zeros(10)
    for i in range(k):
        knn_labels[labels[top_indexes[i]]] += 1

    top_label = argmax(knn_labels)



    # return the label of the test data
    return top_label
