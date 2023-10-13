import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


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

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def Ridge_fit_beta(X, z, lmd):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + lmd * I) @ X.T @ z
    return beta


def lasso_regression(X, z, lmd, max_iter=100000, tol=1e-2):
    lasso = Lasso(lmd, tol=tol, max_iter=max_iter)
    lasso.fit(X, z)
    return lasso

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
        plt.savefig("figures/%s.png" %(savename))

def compare_3d(x, y, z, noise, deg, lmb, name_add="", std=0, mean=0, azim=-120):

    z_true = (z*std + mean  - noise)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z, test_size=0.2)

    X_train = create_design_matrix(x_train, y_train, deg)
    X_test = create_design_matrix(x_test, y_test, deg)
    beta = Ridge_fit_beta(X_train, z_train, lmb)
    z_pred_ridge = (X_test @ beta)*std + mean

    X_train = create_design_matrix(x_train, y_train, deg)
    X_test = create_design_matrix(x_test, y_test, deg)
    lasso = lasso_regression(X_train, z_train, lmb, max_iter=int(1e3), tol=1e-4)
    z_pred_lasso = lasso.predict(X_test)*std + mean

    X_train = create_design_matrix(x_train, y_train, deg)
    X_test = create_design_matrix(x_test, y_test, deg)
    beta = OLS_fit_beta(X_train, z_train)
    z_pred_OLS =  (X_test @ beta)*std + mean

    plot_3d_trisurf(x_test, y_test, z_test*std + mean , azim=azim, title="Test data")
    plt.savefig("figures/test_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_train, y_train, z_train*std + mean , azim=azim, title="Train data")
    plt.savefig("figures/train_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x.ravel(), y.ravel(), z_true, azim=azim, title="Actual data")
    plt.savefig("figures/actual_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_ridge, azim=azim, title="Ridge predict")
    plt.savefig("figures/ridge_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_lasso, azim=azim, title="Lasso predict")
    plt.savefig("figures/lasso_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_OLS, azim=azim, title="OLS predict")
    plt.savefig("figures/ols_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plt.show()


# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')
#print(np.shape(terrain))
n = 100
terrain = terrain[:n, :n]

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()


#noise = np.random.normal(0, 0.1, n*n)  # Generate 2D noise
z = terrain.ravel()

mean_scale = np.mean(z)
std_scale = np.std(z)
z = (z - mean_scale) / std_scale  # Standard scale



lmb = 1e-10
deg = 20


compare_3d(x, y, z, 0, deg, lmb, name_add="n100", std=std_scale, mean=mean_scale, azim=-120)
