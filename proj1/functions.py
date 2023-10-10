from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from imageio import imread
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.linear_model import Lasso, Ridge
import matplotlib as mpl
from matplotlib.cm import ScalarMappable



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from functions import *
def make_data(n, noise_std, seed=1, terrain=False):
    """
    Make data z=f(x)+noise for n steps and normal distributed
    noise with standard deviation equal to noise_std
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(x, y)

    noise = np.random.normal(0, noise_std, size=(n+1,n+1))
    z = FrankeFunction(x, y) + noise
    return x, y, z.ravel()

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def design_matrix(x, y, degree):
    """
    Setting up design matrix with dependency on x and y for a chosen degree
    [x,y,xy,x²,y²,...]
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def mean_scaler(*args):
    """
    Scales arguments by subtracting the mean
    Returns argument followed by its mean
    """
    for arg in args:
        arg -= np.mean(arg, axis=0)

    return args

def OLS(X, z):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    A = np.linalg.pinv(X.T @ X)
    beta = A @ X.T @ z
    return beta

def ridge_regression(X, z, lamda):
    """
    Manual function for ridge regression to find beta
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - lamda:    chosen lamda for the Ridge regression
    returns:
    - beta
    """
    N = X.shape[1]
    beta = np.linalg.pinv(X.T @ X + lamda*np.eye(N)) @ X.T @ z
    return beta

def lasso_regression(X, z, lamda, max_iter=int(1e2), tol=1e-2):
    """
    Sklearns function for lasso regression to find beta
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - lamda:    chosen lamda for the lasso regression
    returns:
    - beta
    """
    lasso = Lasso(lamda, tol=tol, max_iter=max_iter)
    lasso.fit(X, z)
    return lasso

def MSE(data, model):
    """
    takes in actual data and modelled data to find
    Mean Squared Error
    """
    MSE = mean_squared_error(data.ravel(), model.ravel())
    return MSE

def R2(data, model):
    """
    takes in actual data and modelled data to find
    R2 score
    """
    R2 = r2_score(data.ravel(), model.ravel())
    return R2

def plot_3D(x, y, z):
    """
    Takes in:
    x, y: Meshgrid matching data
    z: data

    plots a surface
    """
    #3D plot of predicted values
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_3d_trisurf(x, y, z, scale_std=1, scale_mean=0, savename=None, azim=110, title=""):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)
    surf = ax.plot_trisurf(x, y, z*scale_std + scale_mean, cmap=cm.coolwarm, linewidth=0.2, antialiased=False)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylabel(r"$y$")
    ax.view_init(azim=azim)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    if savename != None:
        plt.savefig("../figures/%s.png" %(savename))


def bootstrap(X_train, X_test, z_train, z_test, n_B, method, lamda=1, max_iter =100):
    z_pred = np.zeros((len(z_test), n_B))
    for i in range(n_B):
        X_, z_ = resample(X_train, z_train)
        if method == "OLS":
            beta = OLS(X_, z_)
            z_pred[:, i] = (X_test @ beta).ravel()

        if method == "RIDGE":
            beta = ridge_regression(X_, z_, lamda)
            z_pred[:, i] = (X_test @ beta).ravel()

        if method == "LASSO":
            model = lasso_regression(X_, z_, lamda, max_iter)
            z_pred[:, i] = model.predict(X_test).ravel()

    return z_pred



def cross_validation(X, z, k_folds, lamda=0, method="RIDGE", max_iter=int(1e4), scale=False):
    """
    Manual algorithm for cross validation using chosen regression method
    to find MSE
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - k_folds:  number of k_folds in the cross validation algorithm
    - lamda:    chosen lamda for the Ridge regression
    - method:   Regression method
    Returns:
    - MSE as a mean over the MSE returned by the cross validation function
    """

    k_fold = KFold(n_splits = k_folds, shuffle=True)

    i = 0
    mse = np.zeros(k_folds)

    for train_idx, test_idx in k_fold.split(X):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        z_train = z[train_idx]
        z_test = z[test_idx]

        if scale == True:
            X_train -= np.mean(X_train, axis=0)
            X_test -= np.mean(X_test, axis=0)
            z_train -= np.mean(z_train, axis=0)
            z_test -= np.mean(z_test, axis=0)

        if method == "OLS":
            beta = OLS(X_train, z_train)
            z_pred = (X_test @ beta)

        if method == "RIDGE":
            beta = ridge_regression(X_train, z_train, lamda)
            z_pred = (X_test @ beta)

        if method == "LASSO":
            model = lasso_regression(X_train, z_train, lamda, max_iter)
            z_pred = model.predict(X_test)

        mse[i] = mean_squared_error(z_test, z_pred)
        i += 1
    return np.mean(mse)

def MSE_R2_beta_degree(x, y, z, degree_max):
    """
    Takes in:
    x, y, z: dataset
    degree_max: maximum polynomial degree for comparison

    Returning:
    arrays of MSE (train and test), R2 and beta computed for
    different polynomial degrees
    """

    MSE_train = np.zeros(degree_max)
    MSE_test = np.zeros(degree_max)
    beta_OLS = np.zeros((degree_max, int((degree_max+1)*(degree_max+2)/2)))
    R2 = np.zeros(degree_max)

    for degree in range(1, degree_max+1):
        X = design_matrix(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train_scaled = X_train, X_train_mean, z_train_scaled, z_train_mean = mean_scaler(X_train, z_train)
        X_test_scaled = X_test - X_train_mean

        n_variables = int((degree+1)*(degree+2)/2)

        beta_OLS[degree-1, 0:n_variables] = OLS(X_train_scaled, z_train)
        zpredict_test = X_test_scaled @ beta_OLS[degree-1,0:n_variables] + z_train_mean#+ np.mean(z_test, axis=0)
        zpredict_train = X_train_scaled @ beta_OLS[degree-1,0:n_variables] + z_train_mean

        MSE_train[degree-1], R2[degree-1] = MSE_R2(z_train, zpredict_train)
        MSE_test[degree-1], R2[degree-1] = MSE_R2(z_test, zpredict_test)

    return MSE_test, MSE_train, beta_OLS, R2
