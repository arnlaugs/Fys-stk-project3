import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import seaborn as sns
import keras
from keras.models import model_from_json
from matplotlib2tikz import save as tikz_save


# The classes in this dataset are labeled after the index of the desired class in the string below

def load_data(st):
    train_data_path = 'emnist/emnist-%s-train.csv' %st
    test_data_path = 'emnist/emnist-%s-test.csv' %st
    mapping_path = 'emnist/emnist-%s-mapping.txt' %st

    mapping = list(map(chr, np.loadtxt(mapping_path, dtype=np.int)[:,1]))
    n = len(mapping)

    # Read in data (https://arxiv.org/pdf/1702.05373.pd)
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Separate labels from data and convert to numpy arrays
    X_train = (train_data.iloc[:, 1:].values).reshape(len(train_data), 28, 28, 1)
    y_train = train_data.iloc[:, 0].values
    X_test  = (test_data.iloc[:, 1:].values).reshape(len(test_data), 28, 28, 1)
    y_test  = test_data.iloc[:, 0].values

    # Making onehot vectors, y-1 because keras expect first class to be 0
    y_train_onehot, y_test_onehot = to_categorical(y_train-1, n), to_categorical(y_test-1, n)
    del train_data, test_data, y_train, y_test

    return X_train, y_train_onehot, X_test, y_test_onehot, mapping

X_train, y_train, X_test, y_test, mapping = load_data("digits")


# load json and create model
json_file = open('model_digits.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_digits.h5")
print("Loaded model from disk")

loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("Model compiled")

#score = loaded_model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

y_predict = loaded_model.predict(X_test)

print("Predicted!")

test_matrix = np.zeros((len(mapping), len(mapping)), dtype=np.int)


for i in range(len(y_test)):
    test_matrix[int(np.argmax(y_test[i])), int(np.argmax(y_predict[i]))] += 1
np.save("test_matrix_digits.npy", test_matrix)

#test_matrix = np.load("test_matrix_digits.npy")

accuracy = np.zeros(len(mapping))
for i in range(len(test_matrix)):
    accuracy[i] = test_matrix[i,i]/np.sum(test_matrix[i])
    if np.isnan(accuracy[i]):
        accuracy[i] = 0

plt.bar(range(len(mapping)), accuracy)
plt.xticks(np.arange(len(mapping)),mapping)
plt.yticks(np.linspace(0,1,11))
plt.grid(axis = "y", linestyle="--")
plt.ylabel("Accuracy")
tikz_save('accuracy.tex', figureheight="\\figureheight", figurewidth="\\figurewidth")

test_matrix = np.array(test_matrix, dtype= np.float)
for i in range(len(test_matrix)):
    test_matrix[i] = test_matrix[i]/np.sum(test_matrix[i])*100

fig, ax = plt.subplots()
sns.heatmap(test_matrix, annot=True, fmt=".1f", ax=ax, cmap="viridis")
plt.xticks(np.arange(len(mapping))+0.5,mapping)
plt.yticks(np.arange(len(mapping))+0.5,mapping, rotation="horizontal")
plt.ylabel("Expected")
plt.xlabel("Predicted")
tikz_save('heatmap_digits_percent.tex', figure=fig, figureheight="\\figureheight", figurewidth="\\figurewidth")


plt.show()
