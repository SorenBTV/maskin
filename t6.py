import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

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


def k_fold(data, k):
    N = len(data)
    fold_size = N // k
    ind = np.arange(N)
    np.random.shuffle(ind)
    k_fold_ind = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_ind = ind[test_start:test_end]
        train_ind = np.concatenate([ind[:test_start], ind[test_end:]])
        k_fold_ind.append((train_ind, test_ind))
    return k_fold_ind


# Initialize variables
np.random.seed(123)  # Setting a seed for reproducibility
n = 100  # Number of data points
max_degree = 15  # Maximum polynomial degree

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()
