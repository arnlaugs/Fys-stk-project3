import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  svm, metrics
#from keras.utils import to_categorical
class_mapping = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def load_data():
    train_data_path = 'emnist/emnist-letters-train.csv'
    test_data_path = 'emnist/emnist-letters-test.csv'

    # Read in data (https://arxiv.org/pdf/1702.05373.pd)
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Separate labels from data and convert to numpy arrays
    X_train = (train_data.iloc[:, 1:].values).reshape(len(train_data), 28, 28, 1)
    y_train = train_data.iloc[:, 0].values
    X_test  = (test_data.iloc[:, 1:].values).reshape(len(test_data), 28, 28, 1)
    y_test  = test_data.iloc[:, 0].values

    # Making onehot vectors, y-1 because keras expect first class to be 0
    #y_train_onehot, y_test_onehot = to_categorical(y_train-1, 26), to_categorical(y_test-1, 26)
    #del train_data, test_data, y_train, y_test
    print('her')
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
X_train=X_train[:1000]
y_train=y_train[:1000]
print(X_train.shape, y_train.shape)
n_samples=len(X_train)
data=X_train.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)
classifier.fit(data,y_train)
print('2')
expected = X_train
predicted = classifier.predict(X_train)
print('3')
print(expected,predicted)
print(metrics.classification_report(expected, predicted))
