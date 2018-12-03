import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from NeuralNetwork import NeuralNetwork

# The classes in this dataset are labeled after the index of the desired class in the string below
class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def load_data():
    train_data_path = 'emnist/emnist-letters-train.csv'
    test_data_path = 'emnist/emnist-letters-test.csv'

    # Read in data (https://arxiv.org/pdf/1702.05373.pd)
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Separate labels from data and convert to numpy arrays
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test  = test_data.iloc[:, 1:].values
    y_test  = test_data.iloc[:, 0].values

    # Making onehot vectors, y-1 because keras expect first class to be 0
    y_train_onehot, y_test_onehot = to_categorical(y_train-1, 26), to_categorical(y_test-1, 26)
    del train_data, test_data, y_train, y_test

    return X_train, y_train_onehot, X_test, y_test_onehot


def view_image(X, y, img_num):
    """
    Image has to be transposed and reshaped to be viewed.
	Parameters
	-----------
	X : array_like, shape=[n_samples, n_features]
		Data
    y : array_like, shape=[n_samples]
        Lables
	"""
    img = np.transpose(X[img_num].reshape(28,28))
    label = class_mapping[np.argmax(y[img_num])]
    plt.imshow(img, cmap='Greys_r')
    plt.title('Label: %s' %label)
    plt.show()

X_train, y_train, X_test, y_test = load_data()


# Defining variables need in the Neural Network
epochs = 100
batch_size = 100
n_hidden_neurons = 10
n_categories = 26

view_image(X_train, y_train, 73)

NN = NeuralNetwork(X_train, y_train, epochs=epochs, batch_size=batch_size,
                   n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)

NN.heatmap_eta_lambda(X_test, y_test)
#NN.heatmap_neurons_eta(X_test, y_test)
