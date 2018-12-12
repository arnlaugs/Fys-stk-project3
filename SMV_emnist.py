import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  svm, metrics
from sklearn.model_selection import train_test_split


class_mapping = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'
labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#labels=['0','1','2','3','4','5','6','7','8','9']
def load_data():
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
    fig, ax = plt.subplots()
    cm=metrics.confusion_matrix(expected, predicted)
    #cm_1=np.zeros((len(expected),len(predicted)))
    #for i in range(len(cm)):

        #t=np.count_nonzero(expected == i)
        #for j in range(len(cm)):

            #p=float(cm[i][j])/float(t)*100
            #cm_1[i][j]=p
            #cm[i][j]=p
            #print(cm_1[i][j],p, type(p), type(cm_1[i][j]))
    plt.imshow(cm,aspect='auto')
    print(cm.shape, len(labels))
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(np.arange(len(labels)),labels,fontsize=14)
    plt.yticks(np.arange(len(labels)),labels,fontsize=14)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i][j]<10:
                text_s='%01.0f' % (cm[i][j])
            else:
                text_s='%02.0f' % (cm[i][j])

            if cm[i][j]>500:
                text = ax.text(j, i, text_s, ha="center", va="center", color="k",fontsize=14)

            else:
                text = ax.text(j, i, text_s, ha="center", va="center", color="w",fontsize=14)
    plt.ylabel('Expected',fontsize=14)
    plt.xlabel('Predicted',fontsize=14)
    print('er')
    plt.savefig("cm_digits_test_.pdf")
    plt.show()

def find_images(test_samples, expected,predicted,labels):
    L=True; I=True; Q=True; G=True
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
#X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2)
train_samples=len(X_train)
test_samples=len(X_test)
#print(len(X_train),test_samples)
#train_samples=5000

#test_samples=500
#test_samples_sum=test_samples+train_samples

#y_test=y_train[train_samples:test_samples_sum]
#X_test=X_train[train_samples:test_samples_sum]
#X_train=X_train[:train_samples]
#y_train=y_train[:train_samples]
#X_test=X_test[:test_samples]
#y_test=y_test[:test_samples]

data=X_train
data_test=X_test


classifier = svm.SVC(degree=2,kernel='poly')
classifier.fit(data,y_train)

expected = y_test
predicted = classifier.predict(data_test)
expected_train=y_train
predicted_train=classifier.predict(data)


print('her')
find_images(test_samples, expected,predicted,labels)



print("accuracy test= ",metrics.accuracy_score(expected, predicted))
print("accuracy train= ",metrics.accuracy_score(expected_train, predicted_train))

#plot_heatmap(expected,predicted,labels)
