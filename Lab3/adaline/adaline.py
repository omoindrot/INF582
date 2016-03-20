from numpy import *
from random import *

# implements the unit step function
unit_step = lambda x: 0 if x < 0 else 1


def training_adaline(w, n, trainingImages, trainingLabels, j, eta):
    # ====================== ADD YOUR CODE HERE =============================
    #  Implement the training step of the adaline algorithm.
    #  return the weight vector w
    for i in range(n):
        index = randint(0, trainingImages.shape[0]-1)
        x = trainingImages[index]
        label = trainingLabels[index]

        y = (label == j)

        C_x = unit_step(w.dot(x))

        w -= eta*(C_x - y)*x

    return w


def classify_adaline(testImages, w):
    # ====================== ADD YOUR CODE HERE =============================
    # Implement the classification step of the adaline algorithm.
    classes = w.dot(testImages)

    class_perceptron = argmax(classes)

    # return the predicted class of the test image
    return class_perceptron