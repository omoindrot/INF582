from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


# ================================ Step 1: Loading and Visualizing Data ================================
print "\nLoading and visualizing Data ...\n"

#Reading of the dataset
# You are free to reduce the number of samples retained for training, in order to reduce the computational cost
#size_training = 60000     # number of samples retained for training
size_training = 2000     # number of samples retained for training
#size_test = 10000     # number of samples retained for testing
size_test = 500     # number of samples retained for testing
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)


# ================================ Step 2: Setting up Neural Network Structure &  Initialize NN Parameters ================================
print "\nSetting up Neural Network Structure ...\n"

# Setup the parameters you will use for this exercise
input_layer_size   = 784        # 28x28 Input Images of Digits
num_labels         = 10         # 10 labels, from 0 to 9 (one label for each digit) 

num_of_hidden_layers = 1
layers = [input_layer_size, 50]

layers = layers + [num_labels]

print "\nInitializing Neural Network Parameters ...\n"

# ================================ DONE ================================
# Fill the randInitializeWeights.py in order to initialize the neural network weights. 
Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

# ================================ Step 9: Training Neural Networks & Prediction ================================
print "\nTraining Neural Network... \n"

#  You should also try different values of the regularization factor
lambd = 0.001
num_iterations = 200
res = fmin_l_bfgs_b(costFunction, nn_weights, fprime=backwards, args=(layers, images_training, labels_training, num_labels, 1.0), maxfun = num_iterations, factr = 1., disp = True)
Theta = roll_params(res[0], layers)


print "\nTesting Neural Network... \n"

pred = predict(Theta, images_test)
print '\nAccuracy: ' + str(mean(labels_test==pred) * 100)

