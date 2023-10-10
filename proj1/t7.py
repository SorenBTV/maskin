import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def lamda_degree_MSE(x, y, z, method, std, n_lmb = 50, maxdegree = 15, k_folds = 5, max_iter = 100, save=True, lmb_min=-12, lmb_max=-1):
    """
    Function to find best degree and lambda parameter
    for the chosen regression method
    takes in:
    - x:                meshrgrid containing x-values
    - y:                meshrgrid containing y-values
    - z:                data
    - method:           Regression method
    - std:              standard deviation of noise added to data
    - n_lmb (opt):      number of lambdas to test for logspace (lmb_min, lmb_max)
    - maxdegree (opt):  maximum degree to test the regression
    - k_folds (opt):    number of kfolds for cross validation method
    - max_iter (opt):   maximum number of iterations used for lasso prediction
    - save (opt):       if true saves ploted heatmap.
    - lmb_min (opt):    minimum power of 10 for lambda (10^(lmb_min))
    - lmb_max (opt):    maximum power of 10 for lambda  (10^(lmb_max))
    returns:
    - optimal lamda, degree and the mse for
    """

    degree = np.arange(1, maxdegree+1)
    lamda = np.logspace(lmb_min, lmb_max, n_lmb)

    if method == "RIDGE" or method == "LASSO":
        degree, lamda = np.meshgrid(degree,lamda)
        mse = np.zeros(np.shape(degree))

        for i in range(maxdegree):
            X = design_matrix(x, y, degree[0, i])
            for j in range(n_lmb):
                mse[j, i] = cross_validation(X, z, k_folds, lamda[j, i], method, max_iter)
            print("\n\n\n ---DEGREE---- %i\n\n\n" %(i))

    elif method == "OLS":
        mse = np.zeros(np.shape(degree))
        for i in range(maxdegree):
            X = design_matrix(x, y, degree[i])
            mse[i] = cross_validation(X, z, k_folds, method=method)

    argmin = np.unravel_index(np.argmin(mse), mse.shape)

    print("---%s---" %(method))
    print("Degree of lowest MSE for %i kfolds" %(k_folds), degree[argmin])
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if method != "OLS":
        print("Lambda of lowest MSE for %i kfolds" %(k_folds), lamda[argmin])
        plt.contourf(degree, lamda, mse, 50, cmap="RdBu")
        plt.colorbar(label=r"$MSE$")
        plt.ylabel(r"$\lambda$")
        plt.yscale("log")
        plt.scatter(degree[argmin], lamda[argmin], marker="x", s=80, label=r"min MSE: %.3f, Lambda: %.2e" %(mse[argmin], lamda[argmin]))
        plt.legend(fontsize=12)

    else:
        plt.plot(degree, mse, "--o", fillstyle="none")
        plt.ylabel(r"$MSE$")
        plt.scatter(degree[argmin], mse[argmin], color="k", marker="x", s=80, label="min MSE: %.3f" %(mse[argmin]))
        plt.legend()

    plt.xlabel("Degree")
    plt.grid(True)

    if save:
        plt.savefig("../figures/best_lambda_%s_0%i.png" %(method, std*10), dpi=300, bbox_inches='tight')
    plt.show()

    if method == "OLS":
        return lamda[0], degree[argmin], mse[argmin]
    else:
        return lamda[argmin], degree[argmin], mse[argmin]


def compare_3d(x, y, z, noise, deg_ols, lmb_ridge, deg_ridge, lmb_lasso, deg_lasso, name_add="franke", std=1, mean=0, azim=50):
    """
    Plots 3D surface for OLS, RIDGE and LASSO regression for the chosen degrees
    and lambdas. Saves the files giving a total of 6 plots.
    takes in:
    - x:            meshrgrid containing x-values
    - y:            meshrgrid containing y-values
    - z:            data with added noise
    - noise:        added noise of data
    - deg_...:      Degrees to plot for the different regressions
    - lmb_...:      Lambdas to use for plot for lasso and ridge
    - name_add:     string to add at end of saved filenames
    - std (opt):    std used to reduce standard scale of z
    - mean (opt):   mean used to reduce standard scale of
    - azim (opt):   azim in degrees for initial position of 3D plot

    plots:
    - 6 plots with surfaces.
    Test and train data, true data and, ridge, ols and lasso regressions
    """
    z_true = (z*std + mean  - noise)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z, test_size=0.2)

    X_train = design_matrix(x_train, y_train, deg_ridge)
    X_test = design_matrix(x_test, y_test, deg_ridge)
    beta = ridge_regression(X_train, z_train, lmb_ridge)
    z_pred_ridge = (X_test @ beta)*std + mean

    X_train = design_matrix(x_train, y_train, deg_lasso)
    X_test = design_matrix(x_test, y_test, deg_lasso)
    lasso = lasso_regression(X_train, z_train, lmb_lasso, max_iter=int(1e3), tol=1e-4)
    z_pred_lasso = lasso.predict(X_test)*std + mean

    X_train = design_matrix(x_train, y_train, deg_ols)
    X_test = design_matrix(x_test, y_test, deg_ols)
    beta = OLS(X_train, z_train)
    z_pred_OLS =  (X_test @ beta)*std + mean

    plot_3d_trisurf(x_test, y_test, z_test*std + mean , azim=azim, title="Test data")
    plt.savefig("../figures/test_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_train, y_train, z_train*std + mean , azim=azim, title="Train data")
    plt.savefig("../figures/train_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x.ravel(), y.ravel(), z_true, azim=azim, title="Actual data")
    plt.savefig("../figures/actual_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_ridge, azim=azim, title="Ridge predict")
    plt.savefig("../figures/ridge_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_lasso, azim=azim, title="Lasso predict")
    plt.savefig("../figures/lasso_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_OLS, azim=azim, title="OLS predict")
    plt.savefig("../figures/ols_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plt.show()



def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Custom Ridge regression function
def Ridge_fit_beta(X, y, alpha):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

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
                beta = Ridge_fit_beta(X_train, z_train, lamda)
                z_pred = (X_test @ beta)

            if method == "LASSO":
                model = lasso_regression(X_train, z_train, lamda, max_iter)
                z_pred = model.predict(X_test)

            mse[i] = mean_squared_error(z_test, z_pred)
            i += 1
    return np.mean(mse)


def compare_beta_lambda(x, y, z, lamda):
    """
    Function to plot how beta parameters react to different values of
    lamda for a design matrix of degree 5

    takes in:
    - x:                meshrgrid containing x-values
    - y:                meshrgrid containing y-values
    - z:                data
    - lamda :           numpy 1D array of lambdas to plot for
    """
    X = create_design_matrix(x, y, 5)
    beta_ridge = np.zeros((len(lamda), X.shape[1]))
    beta_lasso = np.zeros((len(lamda), X.shape[1]))
    i=0
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    for lmb in lamda:
        beta_ridge[i] = ridge_regression(X, z, lmb)
        beta_lasso[i] = lasso_regression(X_train, z_train, lmb, max_iter=10000).coef_
        i +=1

    plt.plot(lamda, beta_ridge)
    plt.title(r"Ridge $\beta$ for degree of 5")
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    #plt.savefig("../figures/ridge_beta.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(lamda, beta_lasso)
    plt.title(r"Lasso $\beta$ for degree of 5")
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    #plt.savefig("../figures/lasso_beta_test.png", dpi=300, bbox_inches='tight')
    plt.show()


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# Initialize variables
np.random.seed(123)  # Setting a seed for reproducibility
n = 25  # Number of data points
max_degree = 6  # Maximum polynomial degree

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()
data = np.column_stack((x, y))
#x = np.random.rand(n)
#y = 3*x**2 + np.random.randn(n)

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n * n)
#z = z + noise

# Lambda values
nlambdas = 11
lambdas = np.logspace(-10, 1, nlambdas)
degree = np.arange(1, max_degree+1)
#degree, lambdas = np.meshgrid(degree, lambdas)

k = 10
"""
mse = np.zeros(np.shape(degree))

for i in range(max_degree):
    X = create_design_matrix(x, z, degree[0,i])
    for j in range(nlambdas):
        mse[j,i] = cross_validation(X, z, k, lambdas[j,i], "RIDGE", int(1e4))
    #print("\n\n\n ---DEGREE---- %i\n\n\n" %(i))
"""

#compare_beta_lambda(x, y, z, lambdas)
