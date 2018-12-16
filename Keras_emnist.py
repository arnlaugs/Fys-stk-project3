import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


# The classes in this dataset are labeled after the index of the desired class in the string below
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
    y_train_onehot, y_test_onehot = to_categorical(y_train-1, 26), to_categorical(y_test-1, 26)
    del train_data, test_data, y_train, y_test

    return X_train, y_train_onehot, X_test, y_test_onehot

X_train, y_train, X_test, y_test = load_data()


batch_size = 128
num_classes = 26
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# Create model. Sequential means that we can add layer by layer.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

"""
# Add layers to the model
# Conv2D are convolutionsal layers that will deal with the input images
# 64 and 32 is the number of neurons , kernel=3 size means we will have a 3x3
# filter matrix. Flatten gives a connection between the convolution and dense
# layers. Dense is the output layer.
model.add(Conv2D(784, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(26, activation='softmax'))


# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

# Training the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
"""
