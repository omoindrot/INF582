import numpy as np
import scipy as sp
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """

    # Project test data onto the LDA space defined by W
    projected_data = np.dot(X, W)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.
    
    datanum, dim = projected_data.shape  # dimensions of the dataset
    label = np.zeros(datanum)
    for i in range(datanum):
        distances = [np.linalg.norm(projected_data[i]-mu) for mu in projected_centroid]
        label[i] = np.argmin(distances)

    # =============================================================

    # Return the predicted labels of the test data X
    return label
