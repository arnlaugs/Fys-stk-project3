import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split


class_mapping = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#labels=['0','1','2','3','4','5','6','7','8','9']
def load_data():
    """ Loads data from the data set, returns x and y for both test and train """
    # Letters
    train_data_path = 'emnist/emnist-letters-train.csv'
    test_data_path = 'emnist/emnist-letters-test.csv'

    #Digits
    #train_data_path = 'emnist/emnist-digits-train.csv'
    #test_data_path = 'emnist/emnist-digits-test.csv'

    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Separate labels from data and convert to numpy arrays
    X_train = (train_data.iloc[:, 1:].values)
    y_train = train_data.iloc[:, 0].values
    X_test  = (test_data.iloc[:, 1:].values)
    y_test  = test_data.iloc[:, 0].values

    return X_train, y_train, X_test, y_test

def plot_heatmap(expected,predicted,labels):

    """Plots a confusion matrix for expected and predicted values. """
    fig, ax = plt.subplots()
    cm=metrics.confusion_matrix(expected, predicted)
    plt.imshow(cm,aspect='auto')
    print(cm.shape, len(labels))
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(np.arange(len(labels)),labels,fontsize=14)
    plt.yticks(np.arange(len(labels)),labels,fontsize=14)
    #Add text
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i][j]<10:
                text_s='%01.0f' % (cm[i][j])
            else:
                text_s='%02.0f' % (cm[i][j])

            if cm[i][j]>500: #black is most sutable on yellow background
                text = ax.text(j, i, text_s, ha="center", va="center", color="k",fontsize=14)

            else:#white is most sutable on purple background
                text = ax.text(j, i, text_s, ha="center", va="center", color="w",fontsize=14)
    plt.ylabel('Expected',fontsize=14)
    plt.xlabel('Predicted',fontsize=14)
    plt.savefig("cm_digits_test_.pdf")
    plt.show()

def find_images(test_samples, expected,predicted,labels):
    """Creats images of misclassified letters. Saves the images. It is costumized to find L,I,Q and G,
    future work will be to make it more general """
    L=True; I=True; Q=True; G=True # It stops after finding one image of each class
    for i in range(test_samples):
        if expected[i]!=predicted[i] and expected[i]==12 and predicted[i]==9 and L==True:
            plt.figure()
            plt.title('Expected: ' + labels[11] + ', Predicted: ' + labels[8], fontsize=14)
            print(i,data_test[i, 1:].shape)
            img_flip = np.transpose(data_test[i, 0:].reshape(28, 28), axes=[1,0])
            plt.imshow(img_flip, cmap='Greys_r')
            plt.savefig('L_I.pdf')
            L=False
        if expected[i]!=predicted[i] and expected[i]==9 and predicted[i]==12 and I==True:
            print('her')
            plt.figure()
            plt.title('Expected: ' + labels[8] + ', Predicted: ' + labels[11], fontsize=14)
            print(i,data_test[i, 1:].shape)
            img_flip = np.transpose(data_test[i, 0:].reshape(28, 28), axes=[1,0])
            plt.imshow(img_flip, cmap='Greys_r')
            plt.savefig('I_L.pdf')
            I=False
        if expected[i]!=predicted[i] and expected[i]==7 and predicted[i]==17 and G==True:
            plt.figure()
            plt.title('Expected: ' + labels[6] + ', Predicted: ' + labels[16], fontsize=14)
            print(i,data_test[i, 1:].shape)
            img_flip = np.transpose(data_test[i, 0:].reshape(28, 28), axes=[1,0])
            plt.imshow(img_flip, cmap='Greys_r')
            plt.savefig('G_Q.pdf')
            G=False
        if expected[i]!=predicted[i] and expected[i]==17 and predicted[i]==7 and Q==True:
            plt.figure()
            plt.title('Expected: ' + labels[16] + ', Predicted: ' + labels[6], fontsize=14)
            print(i,data_test[i, 1:].shape)
            img_flip = np.transpose(data_test[i, 0:].reshape(28, 28), axes=[1,0])
            plt.imshow(img_flip, cmap='Greys_r')
            plt.savefig('Q_G.pdf')
            Q=False



X_train, y_train, X_test, y_test = load_data()

train_samples=len(X_train)
test_samples=len(X_test)

data=X_train
data_test=X_test

#Use suport vector machine to predict.
classifier = svm.SVC(degree=2,kernel='poly')
classifier.fit(data,y_train)

expected = y_test
predicted = classifier.predict(data_test)
expected_train=y_train
predicted_train=classifier.predict(data)

find_images(test_samples, expected,predicted,labels)
print("accuracy test= ",metrics.accuracy_score(expected, predicted))
print("accuracy train= ",metrics.accuracy_score(expected_train, predicted_train))
plot_heatmap(expected,predicted,labels)
