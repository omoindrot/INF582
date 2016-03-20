from finalTest import finalTest

size_training = 60000
size_test = 10000
hidden_layers = [256, 128, 32, 16]
lambd = 3.0
num_iterations = 500
# accuracy: 98.28% !

finalTest(size_training, size_test, hidden_layers, lambd, num_iterations)

print "lambda:", lambd