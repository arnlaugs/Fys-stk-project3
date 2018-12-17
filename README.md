# FYS-STK4155 - Applied data analysis and machine learning: Project 3

In this repository you will find codes used for project 3 in FYS-STK4155. 

### Authors
The projects found in this repository is a results of the collaboration between

* **Maren Rasmussen** - (https://github.com/mjauren)

* **Arnlaug Høgås Skjæveland** - (https://github.com/arnlaugs)

* **Markus Leira Asprusten** - (https://github.com/maraspr)

## Downloding data:
To run thees programs you need to downlode the EMNIST dataset found here: https://www.kaggle.com/crawford/emnist

## Description of programs

**SMV_emnist.py** 

Loads the EMNIST data and trains using Scikit-learn's support vector machine with polynomial kernal. 

**Keras_emnist.py**

Loads the EMNIST data and trains using a Keras convolution neural network. Saves the architecture and weights of the system.

**load_conv.py**

Loads the configuration from *Keras_emnist.py* and makes figures based on the results of the test data.

**NeuralNetwork.py**

Simple multilayer perceptron neural network for classification with one hidden layer. 
Input variables:
* X_data: dataset, features
* Y_data: classes
* n_hidden_neurons: number neurons in the hidden layer
* n_catogories: number of categories / neurons in the final
            output layer
* epochs: number of times running trough training data
* batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
* eta: learning rate
* lmbd: regularization parameter
* activation_func: activation function, sigmoid is standard
* activation_func_out: activation function for output
* cost_func: Cost function
* leaky_a: Slope for negative values in Leaky ReLU

Used by calling:

    NN = NeuralNetwork(X_train, Y_train, ... )
    NN.train() 
    NN.predict(X_test)

Can also provide a heatmap illustrating which values of the learning rate and the number of hidden neurons that gives the best accuracies and a confusion matrix for letters or digits:

    NN = NeuralNetwork(X_train, Y_train, ... )

    NN.heatmap_neurons_eta()
    NN.heatmap_confusion(X_test, y_test, class_mapping)
 
**NN_emnist.py**

Loads the EMNIST data and trains the multilayer perceptron neural network from *NeuralNetwork.py*.

   
