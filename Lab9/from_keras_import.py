# Import python modules
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
from sklearn import cross_validation

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


max_features = 17430
maxlen = 300  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
vocab_size = 17430



# Part 1: Load data from .csv file
############
with open('movie_reviews.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    # Initialize lists for data and class labels
    data = []
    labels = []
    # For each row of the csv file
    for row in reader:
        # skip missing data
        if row[0] and row[1]:
            data.append(row[0])
            y_label = -1 if row[1] == 'negative' else 1
            labels.append(y_label)


# Part 2: Data preprocessing
############
stopwords = ['br']

# For each document in the dataset, do the preprocessing
for doc_id, text in enumerate(data):
    #
    #  ADD YOUR CODE TO THE NEXT BLOCKS
    # Remove punctuation and lowercase
    # DONE: Add your code here. Store results to a list with name 'doc'
    #
    text = text.lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    doc = tokenizer.tokenize(text)
    # Stopword removal
    # DONE: Add your code here. Store results to a list with name 'doc'
    #
    doc = [word for word in doc if word not in stopwords]
    # Stemming
    # DONE: Add your code here. Store results to a list with name 'doc'
    #
    stemmer = PorterStemmer()
    doc = [stemmer.stem(word) for word in doc]
    # Convert list of words to one string
    doc = ' '.join(w for w in doc).encode('ascii')
    doc = one_hot(doc, vocab_size, split=' ')
    data[doc_id] = doc   # list data contains the preprocessed document


data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(data, labels, test_size=0.4, random_state=1033)


# Model learning and prediction
# TODO: test different learning algorithms

y_train = np.array(labels_train)
y_test = np.array(labels_test)
y_train = (y_train == 1).astype('float32')
y_test = (y_test == 1).astype('float32')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(data_train, maxlen=maxlen)
X_test = sequence.pad_sequences(data_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(max_features, 128, input_length=maxlen),
               name='embedding', input='input')
model.add_node(LSTM(64), name='forward', input='embedding')
model.add_node(LSTM(64, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')

# try using different optimizers and different optimizer configs
model.compile('adam', {'output': 'binary_crossentropy'})

print('Train...')
model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=10)
acc = accuracy(y_test,
               np.round(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))
print('Test accuracy:', acc)
