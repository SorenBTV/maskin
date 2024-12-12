import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns


# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Define a function to fit beta coefficients using Ordinary Least Squares (OLS)
def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)

# Define a function to calculate R-squared (R2) score
def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()  # flatten arrays
    return 1 - np.sum((y_actual - y_model)**2) / np.sum((y_actual - np.mean(y_actual))**2)


# Custom Ridge regression function
def Ridge_fit_beta(X, y, alpha):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta


# Define a function to create a design matrix for polynomial regression
def create_design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x, y = x.ravel(), y.ravel()
    N = len(x)
    num_features = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, num_features))
    col = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, col] = x ** (i - j) * y ** j
            col += 1
    return X


# Define a function for k-fold cross-validation
def k_fold(data, k):
    n_samples = len(data)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.seed(123)
    np.random.shuffle(indices)
    k_fold_indices = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        k_fold_indices.append((train_indices, test_indices))
    return k_fold_indices


# Define the Franke function that generates synthetic data
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


"""
def CostOLS(theta,y,X):
    n = len(y)
    return (1.0/n)*anp.sum((y-X @ theta)**2)
"""

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
