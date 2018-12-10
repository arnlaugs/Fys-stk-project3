import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split
#import keras
#from keras.utils import to_categorical
class_mapping = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#labels=['0','1','2','3','4','5','6','7','8','9']
def load_data():
    #train_data_path = 'emnist/emnist-balanced-train.csv'
    #test_data_path = 'emnist/emnist-balanced-test.csv'
    train_data_path = 'emnist/emnist-letters-train.csv'
    test_data_path = 'emnist/emnist-letters-test.csv'
    #train_data_path = 'emnist/emnist-digits-train.csv'
    #test_data_path = 'emnist/emnist-digits-test.csv'
    #train_data_path = 'emnist/emnist-mnist-train.csv'
    #test_data_path = 'emnist/emnist-mnist-test.csv'
    # Read in data (https://arxiv.org/pdf/1702.05373.pd)
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Separate labels from data and convert to numpy arrays
    X_train = (train_data.iloc[:, 1:].values)#.reshape(len(train_data), 28, 28, 1)
    y_train = train_data.iloc[:, 0].values
    X_test  = (test_data.iloc[:, 1:].values)#.reshape(len(test_data), 28, 28, 1)
    y_test  = test_data.iloc[:, 0].values

    return X_train, y_train, X_test, y_test

def plot_heatmap(expected,predicted,labels):
    fig, ax = plt.subplots()
    cm=metrics.confusion_matrix(expected, predicted)
    cm_1=np.zeros((len(expected),len(predicted)))
    for i in range(len(cm)):

        t=np.count_nonzero(expected == i+1)
        for j in range(len(cm)):

            p=float(cm[i][j])/float(t)*100
            cm_1[i][j]=p
            cm[i][j]=p
            #print(cm_1[i][j],p, type(p), type(cm_1[i][j]))
    plt.imshow(cm,vmin=0,vmax=100)
    print(cm.shape, len(labels))
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(np.arange(len(labels)),labels,fontsize=14)
    plt.yticks(np.arange(len(labels)),labels,fontsize=14)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm_1[i][j]<10:
                text_s='%01.0f' % (cm_1[i][j])
            else:
                text_s='%02.0f' % (cm_1[i][j])

            if cm_1[i][j]>70:
                text = ax.text(j, i, text_s, ha="center", va="center", color="k",fontsize=14)

            else:
                text = ax.text(j, i, text_s, ha="center", va="center", color="w",fontsize=14)
    plt.ylabel('Expected',fontsize=14)
    plt.xlabel('Predicted',fontsize=14)
    print('er')
    plt.savefig("cm_test_0.2")
    plt.show()

X_train, y_train, X_test, y_test = load_data()
X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2)
train_samples=len(X_train)
test_samples=len(X_test)
print(len(X_train),test_samples)
#train_samples=1000

#test_samples=50000
#test_samples_sum=test_samples+train_samples

#y_test=y_train[train_samples:test_samples_sum]
#X_test=X_train[train_samples:test_samples_sum]
#X_train=X_train[:train_samples]

#y_train=y_train[:train_samples]

data=X_train
data_test=X_test

gamma_list=[0.01]

for gamma in gamma_list:
    classifier = svm.SVC(degree=2,kernel='poly')
    #classifier = svm.SVC()
    #kernel='linear'
    #classifier= svm.LinearSVC(class_weight='balanced')
    classifier.fit(data,y_train)

    expected = y_test
    predicted = classifier.predict(data_test)
    expected_train=y_train
    predicted_train=classifier.predict(data)

    #print(expected,predicted)
    accuracy=0
    unaccuracy=0
    #print(predicted,predicted_train )
    for i in range(test_samples):
        if expected[i]==predicted[i]:
            accuracy+=1
            #print(predicted[i],expected[i])
        else:
            #print(expected[i],predicted[i],i)
            unaccuracy+=1

    #print(accuracy, accuracy/test_samples, unaccuracy)
    print("accuracy test= ",metrics.accuracy_score(expected, predicted))
    print("accuracy train= ",metrics.accuracy_score(expected_train, predicted_train))
    #print(metrics.confusion_matrix(expected, predicted))
    plot_heatmap(expected,predicted,labels)
