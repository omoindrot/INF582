from kerasTest import finalTest

size_training = 60000
size_test = 10000
hidden_layers = [1024, 1024, 512, 128]
drop = 0.2
num_iterations = 200
# accuracy of 98.73% !

finalTest(size_training, size_test, hidden_layers, dropout=drop, nb_epoch=num_iterations)
