# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:58:59 2014

@author: jb
"""

#imports
import csv
import numpy
from numpy import append,linalg, real, trace, mean, argmax, sum, eye, array, int8, uint8, zeros, float64, float32, concatenate, transpose, dot,shape
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gaussian import loggaussian, loggmm
from mix_gmm_em import mix_gmm_em
from read_dataset import read_dataset
from math import log
from classifier import classify_gmm

# parameters for the lab (can be changed for the bonus question)
size_features = 8  # number of features retained from the PCA (max: 128)
size_training = 20000  #number of samples retained for training
size_test = 9000  #number of samples retained for testing
K_max = 30  # maximum complexity of the mixture - number of GDs / digit (class)

# arrays to store results
results = zeros((K_max+1, 2))

# reading of the dataset
images, labels_training, images_test, labels_test = read_dataset(size_training, size_test,size_features);

# reading of the features extracted from dataset
features_test = numpy.array(list(csv.reader(open("test_data.csv","rb"),delimiter=','))).astype('float') #loading the PCA features of the test data set
features_training = numpy.array(list(csv.reader(open("training_data.csv","rb"),delimiter=','))).astype('float') #loading the PCA features of the training data set
features_test = features_test[:size_test, :size_features]  #only "size_features" first features are kept for training set
features_training = features_training[:size_training, :size_features]  #only "size_features" first features are kept for test set

# arrays containing the model for the mixture
mean_mix = zeros((10, K_max, size_features))  # mean values for the Gaussians
var_mix = zeros((10, K_max, size_features))  # variance for the Gaussians
alpha_mix = zeros((10, K_max))  # mixture weights for the Gaussians

# array for the Expectation step of the EM
W = zeros((10, K_max, size_training))


# Loop on the complexity of the data
# for K in [1, 3, 5, 8, 10, 20, 30]:
for K in [1, 3, 5, 8, 10, 20]:

    print ('mixture complexity', K)
    # Training step
    for i in range(10):
        print(i)
        W_temp = zeros((K_max, size_training))
        data = features_training[labels_training == i, :]
        mean_mix_temp, var_mix_temp, alpha_mix_temp = mix_gmm_em(data, K)  # return mean, variance and weight for
        mean_mix[i, :K, :] = mean_mix_temp
        var_mix[i, :K, :] = var_mix_temp
        alpha_mix[i, :K] = alpha_mix_temp
        W[i, :, :] = W_temp
        # delete temporary arrays (_i)
        del W_temp
        del mean_mix_temp
        del var_mix_temp
        del alpha_mix_temp
    print('training done')
    # classification for training data
    res_total_train = zeros((10, 10))
    for j in range(0, size_training):
        label_from_classifier = classify_gmm(features_training[j,:],var_mix[:,:,:],mean_mix[:,:,:],alpha_mix[:,:],K)   
        res_total_train[labels_training[j], label_from_classifier] += 1
   
    # classification for test data
    res_total_test = zeros((10, 10))
    for j in range(0, size_test):
        label_from_classifier = classify_gmm(features_test[j,:],var_mix[:,:,:],mean_mix[:,:,:],alpha_mix[:,:],K)  
        res_total_test[labels_test[j],label_from_classifier] = res_total_test[labels_test[j],label_from_classifier] +1
    
    print(trace(res_total_train)/sum(sum(res_total_train))   )
    print(trace(res_total_test)/sum(sum(res_total_test))   )
      
    results[K, 0] = trace(res_total_train)/float(size_training)
    results[K, 1] = trace(res_total_test)/size_test
    
    # plot of the results

plt.plot(results[:, 0], 'ro', results[:, 1], 'go')
plt.axis([0, K_max, 0.6, 0.9])
plt.show()
