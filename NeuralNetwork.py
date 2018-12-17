import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from matplotlib2tikz import save as tikz_save
from sklearn import metrics


class NeuralNetwork:
    """
    Sets up a simple neural network with one hidden layer.

    Input variables:
        - X_data: dataset, features
        - Y_data: classes
        - n_hidden_neurons: number neurons in the hidden layer
        - n_catogories: number of categories / neurons in the final
            output layer
        - epochs: number of times running trough training data
        - batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
        - eta: learning rate
        - lmbd: regularization parameter
        - activation_func: activation function, sigmoid is standard
        - activation_func_out: activation function for output
        - cost_func: Cost function
    """
    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=50,
        n_categories=10,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,
        activation_func = 'relu',
        activation_func_out = 'softmax',
        cost_func = 'cross_entropy'
    ):

        # Setting self values
        self.X_data_full = X_data; self.X_data = X_data
        self.Y_data_full = Y_data; self.Y_data = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        # Setting up activation function
        if activation_func == 'sigmoid':
            self.f = self.sigmoid
            self.f_prime = self.sigmoid_prime
        if activation_func == 'softmax':
            self.f = self.softmax
            self.f_prime = self.softmax_prime
        if activation_func == 'tanh':
            self.f = self.tanh
            self.f_prime = self.tanh_prime
        if activation_func == 'identity':
            self.f = self.identity
            self.f_prime = self.identity_prime
        if activation_func == 'relu':
            self.f = self.ReLU
            self.f_prime = self.ReLU_prime

        # Setting up activation function for the output layer
        if activation_func_out == 'sigmoid':
            self.f_out = self.sigmoid
            self.f_out_prime = self.sigmoid_prime
        if activation_func_out == 'softmax':
            self.f_out = self.softmax
            self.f_out_prime = self.softmax_prime
        if activation_func_out == 'tanh':
            self.f_out = self.tanh
            self.f_out_prime = self.tanh_prime
        if activation_func_out == 'identity':
            self.f_out = self.identity
            self.f_out_prime = self.identity_prime
        if activation_func_out == 'relu':
            self.f_out = self.ReLU
            self.f_out_prime = self.ReLU_prime

        # Setting up cost function
        if cost_func == 'cross_entropy':
            self.C_grad = self.cross_entropy_grad
        if cost_func == 'MSE':
            self.C_grad = self.MSE_grad

        # Initialize wrights and biases
        self.create_biases_and_weights()


    def create_biases_and_weights(self):
        """
        Initialize the weights with random numbers from the standard
        normal distribution. Initialize biases to be arrays with 0.01.
        """
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)*1e-3
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)*1e-3
        self.output_bias = np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        """
        Used for training the network.
        1) Calculates z = W*X + b for hidden layer.
        2) Then calculates the activation function of z, giving a.
        3) Calculates z = W*X + b = W*a + b for output layer.
        4) Calculates the softmax function of the output values giving
            the probabilities.
        """
        self.z_h = np.dot(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.f(self.z_h)

        self.z_o = np.dot(self.a_h, self.output_weights) + self.output_bias
        self.a_o = self.f_out(self.z_o)



    def backpropagation(self):
        """
        1) Computes the error of the output result compared to the acual Y-values.
        2) Computes the propagate error (the hidden layer error).
        3) Computes the gradient of the weights and the biases for the output layer
            and hidden layer.
        4) If a regularization parameter is given, the weights are multiplied with
            this before calculating the output weights and biases.
        """
        error_output = self.C_grad(self.a_o, self.Y_data) * self.f_out_prime(self.z_o)
        error_hidden = np.dot(error_output, self.output_weights.T) * self.f_prime(self.z_h)

        self.output_weights_gradient = np.dot(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.dot(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias    -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias    -= self.eta * self.hidden_bias_gradient


    def train(self):
        """
        Trains the model. For each epoch, for each of the minibatches:
            1) Chose datapoints for minibatch.
            2) Make data of the chosen bathes of datapoints.
            3) Run feed forward and backpropagation
        """
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # Pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # Minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()



    def heatmap_neurons_eta(self, X_test, y_test, save=False):
        """
        Illustrates the accuracy of different combinations
        of learning rates eta and number of neurons in the hidden
        layer in a heatmap.
        """
        sns.set()

        eta_vals = np.logspace(-6, -1, 6)
        neuron_vals = [1,10,100,1000]

        train_accuracy = np.zeros((len(neuron_vals),len(eta_vals)))
        test_accuracy = np.zeros((len(neuron_vals),len(eta_vals)))

        for i, neuron in enumerate(neuron_vals):
            for j, eta in enumerate(eta_vals):
                print("training DNN with %4d neurons and SGD eta=%0.6f." %(neuron,eta) )
                DNN = NeuralNetwork(self.X_data_full, self.Y_data_full, eta=eta,
                                    lmbd=0.0, epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    n_hidden_neurons=neuron,
                                    n_categories=self.n_categories)
                DNN.train()

                train_pred = DNN.predict(self.X_data_full)
                test_pred = DNN.predict(X_test)

                train_accuracy[i][j] = accuracy_score(np.argmax(self.Y_data_full, axis=1), train_pred)
                test_accuracy[i][j] = accuracy_score(np.argmax(y_test, axis=1), test_pred)
                print(test_accuracy[i][j])


        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Training Accuracy")
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("hidden neurons")
        ax.set_xticklabels(eta_vals)
        ax.set_yticklabels(neuron_vals)
        if save:
            tikz_save('heatmap_train.tex', figureheight="\\figureheight", figurewidth="\\figurewidth")
        plt.show()

        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Test Accuracy")
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("hidden neurons")
        ax.set_xticklabels(eta_vals)
        ax.set_yticklabels(neuron_vals)
        if save:
            tikz_save('heatmap_test.tex', figureheight="\\figureheight", figurewidth="\\figurewidth")
        plt.show()


    def heatmap_confusion(self, X_test, y_test, labels):
        """
        Plots a confusion matrix for expected and predicted values.
        """
        self.train()

        predicted = self.predict(X_test)
        expected = np.argmax(y_test, axis=1)

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
        plt.show()



    def feed_forward_out(self, X):
        """
        Feed forward for output. Does the same as feed_forward, but
        does not save the variables. Returns the probabilities.
        """
        z_h = np.dot(X, self.hidden_weights) + self.hidden_bias
        a_h = self.f(z_h)

        z_o = np.dot(a_h, self.output_weights) + self.output_bias
        a_o = self.f_out(z_o)

        return a_o


    def predict(self, X):
        """
        Returns the most probable class.
        """
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)


    """ ACTIVATION FUNCTIONS """
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def softmax(self, z):
        exp_term = np.exp(z)
        return exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def softmax_prime(self, z):
        return self.softmax(z)*(1-self.softmax(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1 - self.tanh(z)**2

    def identity(self, z):
        return z

    def identity_prime(self, z):
        return 1

    def ReLU(self, z):
        return np.maximum(z, 0)

    def ReLU_prime(self, z):
        z[z<=0] = 0
        z[z>0] = 1
        return z


    """ COST FUNCTIONS """
    def MSE_grad(self, a, y):
        return (a - y)

    def cross_entropy_grad(self, a, y):
        return np.nan_to_num((a - y)/(a*(1.0 - a)))
