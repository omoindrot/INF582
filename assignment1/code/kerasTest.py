from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


def finalTest(size_training, size_test, hidden_layers, dropout=0.2, nb_epoch=50, batch_size=128):
    print "\nBeginning of the finalTest... \n"

    images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

    X_train = images_training.reshape(60000, 784)
    X_test = images_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    Y_train = np_utils.to_categorical(labels_training, 10)
    Y_test = np_utils.to_categorical(labels_test, 10)

    # Setup the parameters you will use for this exercise
    input_layer_size = 784        # 28x28 Input Images of Digits
    num_labels = 10         # 10 labels, from 0 to 9 (one label for each digit)

    model = Sequential()

    model.add(Dense(784, input_dim=input_layer_size, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    for layer in hidden_layers:
        model.add(Dense(layer, init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    model.add(Dense(10, init='he_normal'))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    history = model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=2,
              validation_data=(X_test, Y_test))

    plt.figure(1)
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['acc'])
    plt.show()

    score = model.evaluate(images_test, y_test,
                           show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print(history.history['val_acc'])



