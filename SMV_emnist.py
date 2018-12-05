import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  svm, metrics
import keras
from keras.utils import to_categorical
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
    #X_train_onehot, X_test_onehot = to_categorical(X_train-1, 26), to_categorical(X_test-1, 26)
    #del train_data, test_data, y_train, y_test

    #return X_train_onehot, y_train_onehot, X_test_onehot, y_test_onehot
    return X_train, y_train, X_test, y_test
train_samples=10000
test_samples=100

X_train, y_train, X_test, y_test = load_data()
X_train=np.squeeze(X_train)
X_test=np.squeeze(X_test)
X_train=X_train[:train_samples]
X_test=X_test[:test_samples]
y_train=y_train[:train_samples]
y_test=y_test[:test_samples]
print(X_train.shape, y_train.shape)

data=X_train.reshape((train_samples, -1))
data_test=X_test.reshape((test_samples,-1))
print(data.shape)
#gamma_list=[0.001,0.01,0.1,1.0,10]
gamma_list=[0.1]
for gamma in gamma_list:
    #classifier = svm.SVC(gamma=gamma,C=200, decision_function_shape='ovo',cache_size=8000,probability=False,kernel='rbf')
    classifier = svm.SVC(kernel='linear')
    #classifier= svm.LinearSVC()
    classifier.fit(data,y_train)

    expected = y_test
    predicted = classifier.predict(data_test)
    expected_train=y_train
    predicted_train=classifier.predict(data)

    #print(expected,predicted)
    #accuracy=0
    #unaccuracy=0
    """
    for i in range(test_samples):
        if expected[i]==predicted[i]:
            accuracy+=1
            #print(predicted[i],expected[i])
        else:
            unaccuracy+=1
    """
    #print(accuracy, accuracy/test_samples, unaccuracy)
    print("accuracy test= ",metrics.accuracy_score(expected, predicted)," gamma= ", gamma)
    print("accuracy train= ",metrics.accuracy_score(expected_train, predicted_train)," gamma= ", gamma)
