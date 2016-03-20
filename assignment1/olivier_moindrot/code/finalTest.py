from numpy import *
from read_dataset import read_dataset
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from displayData import displayData


def finalTest(size_training, size_test, hidden_layers, lambd, num_iterations):
    print "\nBeginning of the finalTest... \n"

    images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)
    # Setup the parameters you will use for this exercise
    input_layer_size = 784        # 28x28 Input Images of Digits
    num_labels = 10         # 10 labels, from 0 to 9 (one label for each digit)
    layers = [input_layer_size] + hidden_layers + [num_labels]
    num_of_hidden_layers = len(hidden_layers)
    # Fill the randInitializeWeights.py in order to initialize the neural network weights.
    Theta = randInitializeWeights(layers)

    # Unroll parameters
    nn_weights = unroll_params(Theta)
    res = fmin_l_bfgs_b(costFunction, nn_weights, fprime=backwards, args=(layers, images_training, labels_training, num_labels, lambd), maxfun = num_iterations, factr = 1., disp = True)
    Theta = roll_params(res[0], layers)

    print "\nTesting Neural Network... \n"

    pred_training = predict(Theta, images_training)
    print '\nAccuracy on training set: ' + str(mean(labels_training == pred_training) * 100)

    pred = predict(Theta, images_test)
    print '\nAccuracy on test set: ' + str(mean(labels_test == pred) * 100)

    # Display the images where the algorithm got wrong
    temp = (labels_test == pred)
    indexes_false = []
    for i in range(size_test):
        if temp[i] == 0:
            indexes_false.append(i)

    displayData(images_training[indexes_false, :])
