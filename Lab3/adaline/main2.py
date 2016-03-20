from numpy import *
import matplotlib.pyplot as plt
from loadMnist import loadMnist
from adaline import training_adaline, classify_adaline


# Load training and test data
trainingImages, trainingLabels = loadMnist('training')
testImages, testLabels = loadMnist('testing')

# Keep a subset of the training and test data
trainingImages = trainingImages[:50000, :]
trainingLabels = trainingLabels[:50000]

testImages = testImages[:2500, :]
testLabels = testLabels[:2500]

print "Images loaded"

predictedDigits = zeros(testImages.shape[0])

    
w = random.rand(10, 784)
eta = 0.0003
n = 5000
iteration_number = 100
results = zeros(iteration_number)
for k in range(iteration_number):
    for j in range(10):
        w[j, :] = training_adaline(w[j, :], n, trainingImages, trainingLabels, j, eta)
        
    for i in range(testImages.shape[0]):
        predictedDigits[i] = classify_adaline(testImages[i, :], w)

    # Calculate accuracy
    correct = 0
    for i in range(testImages.shape[0]):
        if predictedDigits[i] == testLabels[i]:
            correct += 1
            
    results[k] = correct/float(testImages.shape[0])
    print "Accuracy at iteration", k+1, ": ", results[k]


plt.plot(results[:], 'r--')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.axis([0, iteration_number-1 ,0.65, 0.95])
plt.show()
