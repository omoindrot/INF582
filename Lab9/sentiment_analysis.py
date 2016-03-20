# Import python modules
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import svm


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
stopwords = ['br', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
             'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

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

    # Stop word removal
    # DONE: Add your code here. Store results to a list with name 'doc'
    #
    doc = [word for word in doc if word not in stopwords]

    # Stemming
    # DONE: Add your code here. Store results to a list with name 'doc'
    #
    stemmer = PorterStemmer()
    doc = [stemmer.stem(word) for word in doc]

    # Convert list of words to one string
    doc = ' '.join(w for w in doc)
    data[doc_id] = doc   # list data contains the preprocessed document


# Part 3: Feature extraction and the TF-IDF matrix
#############
#
#  ADD YOUR CODE HERE
#
# Create the TF-IFD matrix as described in the lab assignment
#
#
vect = TfidfVectorizer()
tf_idf = vect.fit_transform(data)
tf_idf = tf_idf.toarray()
print 'Size of TF-IDF matrix: ', tf_idf.shape


# Part 4: Model learning and prediction
#############
# Split the data into random train and test subsets. Here we use 40% of the
# data for testing)
#
#   ADD YOUR CODE HERE
#
# Use the code given in the description to split the dataset
#
data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(tf_idf, labels, test_size=0.4, random_state=42)


# Model learning and prediction
# TODO: test different learning algorithms
#
#
#   ADD YOUR CODE HERE
#
# Initialize the classification model, fit the parameters and predict results
# for the test data (see lab description)
#
model_lr = LogisticRegression()
model_lr.fit(data_train, labels_train)
labels_predicted = model_lr.predict_proba(data_test)

# Evaluation of the prediction
#
#
#   ADD YOUR CODE HERE
#
# Compute precision, recall and F1 score (see description)
#
labels = []
for label in labels_predicted:
    if label[1] > label[0]:
        labels.append(1)
    else:
        labels.append(-1)

print "---- Logistic Regression ----"
print classification_report(labels_test, labels)


# With SVM
clf = svm.SVC(decision_function_shape='ovo', kernel='rbf')
clf.fit(data_train, labels_train)
labels_predicted = clf.predict(data_test)

labels = []
for label in labels_predicted:
    if label[1] > label[0]:
        labels.append(1)
    else:
        labels.append(-1)

print "---- SVM ----"
print classification_report(labels_test, labels)


