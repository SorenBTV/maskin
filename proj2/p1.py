import numpy as np
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from functions import *


def CostOLS(theta,y,X):
    n = len(y)
    return (1.0/n)*anp.sum((y-X @ theta)**2)


def CostRidge(theta, y, X, lmb):
    n = len(y)
    return (1.0/n)*anp.sum((y-X @ theta)**2)+lmb*theta.T@theta

def GD(X, y, M=1, epochs=500, learning_rate=0.001, momentum=0.9, lmb=1e-5, Ridge=False):
    n = len(y)
    m = int(n/M)

    theta = np.random.randn(3,1)
    eta = learning_rate  # Corrected the indentation here

    if Ridge==True:
        theta_linreg = Ridge_fit_beta(X, y, lmb)
        change = 0

        for epoch in range(epochs):
            for i in range(m):
                random_index = M * np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]

                sum = np.sum((yi - xi@theta) * xi)
                gradients = -2 * sum + np.dot(2*lmb, theta) + momentum*change
                change = eta * gradients
                theta -= change
    else:
        theta_linreg = OLS_fit_beta(X,y)
        training_gradient = grad(CostOLS)
        change = 0
        for epoch in range(epochs):
            for i in range(m):
                random_index = M * np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                gradients=(1.0/M)*training_gradient(theta, yi, xi)+momentum*change
                change = eta * gradients
                theta -= change

    xnew = np.linspace(0, 1, n)
    Xnew = np.c_[np.ones((n,1)), xnew, xnew*xnew]
    ypredict = Xnew @ theta
    ypredict2 = Xnew @ theta_linreg
    print(theta_linreg)

    return xnew, ypredict, ypredict2



def SGD(X, y, M=5, epochs=500, learning_rate=0.001, momentum=0.9, lmb=1e-5, Ridge=False):
    n = len(y)
    m = int(n/M)

    theta = np.random.randn(3,1)
    eta = learning_rate
    errors = []

    if Ridge==True:
        theta_linreg = Ridge_fit_beta(X, y, lmb)
        change = 0
        training_gradient = grad(CostRidge)
        for epoch in range(epochs):
            for i in range(m):
                random_index = M * np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                #print("shape of xi", np.shape(xi))
                gradients=(1.0/M)*training_gradient(theta, yi, xi, lmb)+momentum*change
                change = eta * gradients
                theta -= change
    else:
        theta_linreg = OLS_fit_beta(X,y)
        training_gradient = grad(CostOLS)
        change = 0
        for epoch in range(epochs):
            for i in range(m):
                random_index = M * np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                #print("shape of xi", np.shape(theta))
                gradients=(1.0/M)*training_gradient(theta, yi, xi)+momentum*change
                change = eta * gradients
                theta -= change

    xnew = np.linspace(0, 1, n)
    Xnew = np.c_[np.ones((n,1)), xnew, xnew*xnew]
    ypredict = Xnew @ theta
    ypredict2 = Xnew @ theta_linreg
    print(theta_linreg)

    return xnew, ypredict, ypredict2



np.random.seed(123)
n = 100
x = np.linspace(0, 1, n)
X = np.c_[np.ones((100,1)), x, x*x]

"""
poly=10
X = np.zeros(n, 10)
for i in range(poly)
    X[:, i] = x**i
"""

y = 4*X[:,0] + 3*X[:,1] + 2*X[:,2]# + np.random.randn(100, 1)

print("X shapes", np.shape(X))
print("y shape",np.shape(y))
xnew, ypredict, ypredict2 = SGD(X, y)

plt.plot(xnew, ypredict, "b-", label="")
plt.plot(xnew, ypredict2, "k--", label="")
plt.plot(x, y ,'r.')
plt.axis([0,1.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
