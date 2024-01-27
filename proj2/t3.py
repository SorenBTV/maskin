
import numpy as np
from sklearn.datasets import load_breast_cancer
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from functions import *
from activation_functions import *


def get_predictions(a_o):
    return np.argmax(a_o, 0)

def get_accuracy(probabilities, Y):
    print(probabilities, Y)
    return np.sum(probabilities == Y) / Y.size


def GD(X, Y, iterations, eta):
    hidden_weights, hidden_bias, output_weights, output_bias = init_params(n_inputs, n_features, n_hidden_neurons, n_categories)
    for i in range(iterations):
        dWo, dBo, dWh, dBh, probabilities = back_propagation(X, Y)
        hidden_weights = hidden_weights - eta * dWh
        hidden_bias = hidden_bias - eta * dBh
        output_weights = output_weights - eta * dWo
        output_bias = output_bias - eta * dBo

        print("Accuracy: ", get_accuracy(probabilities, Y))

    return hidden_weights, hidden_bias, output_weights, output_bias




def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    #print("X:", X.shape, "hidden_weights:", hidden_weights.shape)
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    #print("z_h:", z_h.shape)
    # activation in the hidden layer
    #a_h = np.matmul(sigmoid(z_h),output_weights)
    a_h = sigmoid(z_h)

    z_h2 = np.matmul(X, hidden_weights2) + hidden_bias2
    a_h2 = sigmoid(z_h)
    #print("a_h:", a_h.shape)
    #print(a_h.shape, output_weights.shape)
    # weighted sum of inputs to the output layer
    #print(a_h.shape, output_weights.shape)
    a_o = np.matmul(a_h, output_weights) + output_bias
    #print("a_o:", a_o.shape)

    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    probabilities = sigmoid(a_o)
    #print(probabilities.shape)

    return z_h, a_h, z_h2, a_h2, a_o, probabilities


def back_propagation(X, Y):
    m = Y.size
    #one_hot_Y = one_hot(Y)


    z_h, a_h, z_h2, a_h2, a_o, probabilities = feed_forward(X)
    #print("probabilities:", probabilities.shape, "Y:", Y.shape)
    #error in the output layer
    error_output = probabilities - Y
    #print("err_out:", error_output.shape, "out_weight:", output_weights.shape)
    #error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    error_hidden2 = np.matmul(error_hidden, hidden_weights.T) * a_h2 * (1 - a_h2)
    #print("err_hidden:", error_hidden.shape)
    #gradients for the output layer
    #output_weights_gradient = np.matmul(a_h, error_output)
    dWo = 1 / m * a_h.T.dot(error_output)
    #print("dWo:", dWo.shape)
    #output_bias_gradient = np.sum(error_hidden, axis=0)
    dBo = 1 / m * np.sum(error_output, axis=0)
    #print("dBo:", dBo.shape)

    #hidden_weights_gradient = np.matmul(X.T, error_hidden)
    dWh = 1 / m * X.T.dot(error_hidden)
    #print("dWh:", dWh.shape)
    #hidden_bias_gradient = np.sum(error_hidden, axis=0)
    dBh = 1 / m * np.sum(error_hidden, axis=0)
    #print("dBh:", dBh.shape)
    dWh2 = 1 / m * X.T.dot(error_hidden2)

    dBh2 = 1 / m * np.sum(error_hidden2, axis=0)

    loss = -(1/m) * np.sum(Y*np.log(probabilities) + (1-Y)*np.log(1-probabilities))

    #return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient
    return dWo, dBo, dWh, dBh, dWh2, dBh2, loss






# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)

def hidden_w_b(n_features, n_hidden_neurons):
    hidden_weights = np.random.randn(n_features, n_hidden_neurons)
    hidden_bias = np.zeros(n_hidden_neurons) + 0.01
    return hidden_weights, hidden_bias


def init_params(n_inputs, n_features, n_hidden_neurons, n_hidden_neurons2, n_categories):

    hidden_weights, hidden_bias = hidden_w_b(n_features, n_hidden_neurons)
    hidden_weights2, hidden_bias2 = hidden_w_b(n_features, n_hidden_neurons2)


    # weights and bias in the hidden layer
    #hidden_weights = np.random.randn(n_features, n_hidden_neurons)
    #hidden_bias = np.zeros(n_hidden_neurons) + 0.01

    # weights and bias in the output layer
    output_weights = np.random.randn(n_hidden_neurons, n_categories)
    output_bias = np.zeros(n_categories) + 0.01

    return hidden_weights, hidden_bias, hidden_weights2, hidden_bias2, output_weights, output_bias



# ensure the same random numbers appear every time
np.random.seed(0)
"""
load_data = load_breast_cancer()
data = load_data.data
target = load_data.target
train_size = 0.8
test_size = 1 - train_size

inputs = load_data.data
labels = load_data.target

labels = labels.reshape((labels.shape[0], 1))

X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
"""



# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

"""
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
"""

# The XOR gate
yXOR = np.array( [ 0, 1 ,1, 0])
# The OR gate
yOR = np.array( [ 0, 1 ,1, 1])
# The AND gate
yAND = np.array( [ 0, 0 ,0, 1])

yXOR = yXOR.reshape((yXOR.shape[0], 1))
yOR = yOR.reshape((yOR.shape[0], 1))
yAND = yAND.reshape((yAND.shape[0], 1))


# Defining the neural network
n_inputs, n_features = X.shape
print(n_inputs, n_features)
n_hidden_neurons = 10
n_hidden_neurons2 = 10
n_categories = 1
m = yOR.size

hidden_weights, hidden_bias, hidden_weights2, hidden_bias2, output_weights, output_bias = init_params(n_inputs, n_features, n_hidden_neurons, n_hidden_neurons2, n_categories)

#print("hidden weights:", hidden_weights)
#print("hidden bias:", hidden_bias)
#print("output weights:", output_weights)
#print("output bias:", output_bias)
eta = 0.1
lmb = 0.01

#a_h, prob = feed_forward(X)
#print(prob)
#W1, b1, W2, b2 = GD(X, yOR, iterations=1000, eta=0.01)
loss_list = []
for i in range(1000):
    dWo, dBo, dWh, dBh, dWh2, dBh2, loss = back_propagation(X, yOR)
    print("loss:", loss)
    loss_list.append(loss)
    hidden_weights -= eta * dWh #+ momentum
    hidden_bias -= eta * dBh
    hidden_weights2 -= eta * dWh2
    hidden_bias2 -= eta*dBh2
    output_weights -= eta * dWo
    output_bias -= eta * dBo

plt.plot(loss_list)
plt.show()

test = np.array([0, 1])
z_h, a_h, z_h2, a_h2, a_o, probabilities = feed_forward(test)
if probabilities >= 0.5:
    print(1)
else:
    print(0)















#
