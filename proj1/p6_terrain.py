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

# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Define a function to fit beta coefficients using Ordinary Least Squares (OLS)
def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Define a function to calculate the R-squared score
def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()
    return 1 - np.sum((y_actual - y_model)**2) / np.sum((y_actual - np.mean(y_actual))**2)

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

# Custom Ridge regression function
def Ridge_fit_beta(X, y, alpha):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

# Function to create a design matrix for polynomial regression
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


# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')
#print(np.shape(terrain))
n = 50 # Number of data points
terrain = terrain[:n, :n]

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()


# Standardize the terrain data
z = terrain.ravel()
mean_scale = np.mean(z)
std_scale = np.std(z)
z = (z - mean_scale) / std_scale  # Standard scale

# Initialize a StandardScaler
scaler = StandardScaler()

# Data preprocessing for different regression methods
data = np.column_stack((x, y, z))

# Define the number of lambdas for Ridge and Lasso regression
nlambdas = 6
lambdas = np.logspace(-5, 0, nlambdas)

# Define the maximum polynomial degree and a range of degrees to consider
max_deg = 6
degrees = np.arange(1, max_deg+1)

# Define the number of folds for cross-validation
k = 10



#OLS
plt.figure(figsize=(10, 6))

# Initialize arrays to store error metrics and results and results for each fold
train_mse = np.empty(degrees.shape)
test_mse = np.empty_like(train_mse)
scores_KFold = np.zeros(k)
k_fold_indices = k_fold(data, k)

# Perform cross-validated OLS regression for different polynomial degrees
for degree in degrees:
    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        train, test = data[train_indices], data[test_indices]
        Xtrain = create_design_matrix(train[:,0], train[:,1], degree=degree)
        Xtest = create_design_matrix(test[:,0], test[:,1], degree=degree)
        scaler.fit(Xtrain)
        Xtrain_scaled, Xtest_scaled = scaler.transform(Xtrain), scaler.transform(Xtest)

        beta_train = OLS_fit_beta(Xtrain_scaled, train[:,2])
        beta_test = OLS_fit_beta(Xtest_scaled, test[:,2])

        zpred_train, zpred_test = Xtrain_scaled @ beta_train, Xtest_scaled @ beta_test
    train_mse[degree-1] = np.mean(MSE(train[:,2], zpred_train))
    test_mse[degree-1] = np.mean(MSE(test[:,2], zpred_test))

# Plot the Mean Squared Error for different polynomial degrees with OLS
#plt.plot(degrees, train_mse, ".--", label="Train")
plt.plot(degrees, test_mse, ".-", label="Test")
plt.title("MSE for terrain data using OLS with cross-validation")
plt.grid()
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.savefig("figures/OLS_MSE_terrain_cross_validation.pdf", dpi=300)
#plt.show()

#Ridge regression
plt.figure(figsize=(10, 6))
# Initialize arrays to store results for Ridge regression
test_mse = np.zeros(len(lambdas))
train_mse = np.zeros(len(lambdas))
error = np.zeros((len(degrees), len(lambdas)))
scores_KFold = np.zeros(k)
k_fold_indices = k_fold(data, k)

# Perform cross-validated Ridge regression for different polynomial degrees and lambdas
for i in range(len(lambdas)):
    for j, degree in enumerate(degrees):
        for train_indices, test_indices in k_fold_indices:
            train, test = data[train_indices], data[test_indices]
            Xtrain = create_design_matrix(train[:,0], train[:,1], degree)
            Xtest = create_design_matrix(test[:,0], test[:,1], degree)
            scaler.fit(Xtrain)
            Xtrain_scaled, Xtest_scaled = scaler.transform(Xtrain), scaler.transform(Xtest)

            beta_train = Ridge_fit_beta(Xtrain_scaled, train[:,2], lambdas[i])
            beta_test = Ridge_fit_beta(Xtest_scaled, test[:,2], lambdas[i])

            ztrain_pred = Xtrain_scaled @ beta_train
            ztest_pred = Xtest_scaled @ beta_test
        test_mse[i] = MSE(test[:,2], ztest_pred)
        train_mse[i] = MSE(train[:,2], ztrain_pred)
    error[:, i] = test_mse


# Create a heatmap of the Mean Squared Error for Ridge regression
heatmap = sns.heatmap(error, annot=True, annot_kws={"size": 7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=degrees, cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Polynomial degree")
heatmap.set_xlabel(r'$\lambda$')
heatmap.set_title("MSE heatmap for terrain data using ridge with cross-validation")

# Display and save the heatmap
plt.tight_layout()
plt.savefig("figures/Ridge_MSE_heatmap_terrain_cross_validation.pdf", dpi=300)
#plt.show()



#Lasso

# Initialize arrays to store results for Lasso regression
test_mse = np.zeros(len(lambdas))
train_mse = np.zeros(len(lambdas))
error = np.zeros((len(degrees), len(lambdas)))
scores_KFold = np.zeros(k)
k_fold_indices = k_fold(data, k)
plt.figure(figsize=(10, 6))

# Perform cross-validated Lasso regression for different polynomial degrees and lambdas
for i in range(len(lambdas)):
    print(i)
    for j, degree in enumerate(degrees):
        for train_indices, test_indices in k_fold_indices:
            train, test = data[train_indices], data[test_indices]
            Xtrain = create_design_matrix(train[:,0], train[:,1], degree=degree)
            Xtest = create_design_matrix(test[:,0], test[:,1], degree=degree)
            scaler.fit(Xtrain)
            Xtrain_scaled, Xtest_scaled = scaler.transform(Xtrain), scaler.transform(Xtest)

            lasso = Lasso(alpha=lambdas[i], max_iter=100000)
            lasso.fit(Xtrain_scaled, train[:,2])

            ztrain_pred = lasso.predict(Xtrain_scaled)
            ztest_pred = lasso.predict(Xtest_scaled)

        train_mse[degree-1] = mean_squared_error(train[:,2], ztrain_pred)
        test_mse[degree-1] = mean_squared_error(test[:,2], ztest_pred)
    error[:, i] = test_mse

# Create a heatmap of the Mean Squared Error for Lasso regression
heatmap = sns.heatmap(error, annot=True, annot_kws={"size": 7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=degrees, cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Polynomial degree")
heatmap.set_xlabel(r'$\lambda$')
heatmap.set_title("MSE heatmap for terrain data using lasso with cross-validation")

# Display and save the heatmap
plt.tight_layout()
plt.savefig("figures/Lasso_MSE_heatmap_terrain_cross_validation.pdf", dpi=300)
#plt.show()
