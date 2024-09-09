import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Define a function to fit beta coefficients using Ordinary Least Squares (OLS)
def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

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

# Define the Franke function for generating synthetic data
def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



# Initialize variables
np.random.seed(123)  # Setting a seed for reproducibility
n = 100  # Number of data points
n_bootstraps = 100  # Number of bootstrap samples
max_degree = 15  # Maximum polynomial degree

# Generate random values for x and y within [0, 1]
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)

# Add random noise to the generated data
noise = np.random.normal(0, 0.1, n * n)
z = z + noise

# Create an array of polynomial degrees to consider
degrees = np.arange(1, max_degree+1, 1)

# Initialize arrays to store error metrics
error = np.zeros(max_degree)
error_train = np.zeros(max_degree)
error_test = np.zeros(max_degree)
bias = np.zeros(max_degree)
variance = np.zeros(max_degree)
mse_train = np.empty(n_bootstraps)
mse_test = np.empty(n_bootstraps)

# Loop through different polynomial degrees
for degree in degrees:

    X = create_design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z.flatten(), test_size=0.2)

    z_tilde = np.empty((z_train.shape[0], n_bootstraps))
    z_predict = np.empty((z_test.shape[0], n_bootstraps))


    # Perform bootstrap resampling
    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = OLS_fit_beta(X_, z_)

        z_tilde[:, i] = (X_ @ beta).ravel()
        z_predict[:, i] = (X_test @ beta).ravel()

        mse_train[i] = MSE(z_, z_tilde[:, i])
        mse_test[i] = MSE(z_test, z_predict[:, i])



    # Calculate and store error metrics, bias, and variance
    error_train[degree-1] = np.mean(mse_train)
    error_test[degree-1] = np.mean(mse_test)
    bias[degree-1] = np.mean((z_test - np.mean(z_predict, axis=1))**2)
    variance[degree-1] = np.mean(np.var(z_predict, axis=1))
    #print(f"{error_test[degree]:g} = {bias[degree]+variance[degree]:g}")


#Plotting and saving figures.

# Getting the current directory of the running script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Defining the path to the figures directory
output_dir = os.path.join(current_dir, "..", "figures")

# Plot the Mean Squared Error for different polynomial degrees
plt.figure(figsize=(8, 5))
plt.plot(degrees, error_train,".--", label="Error train")
plt.plot(degrees, error_test, label="Error test")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("MSE for the franke function with bootstrap resampling")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "MSE_bootstrap_franke.pdf"))
#plt.show()

# Plot Bias-Variance trade-off
plt.figure(figsize=(8, 5))
plt.plot(degrees, error_test,".--", label="Error test")
plt.plot(degrees, bias, label="Bias")
plt.plot(degrees, variance, label="Variance")
plt.title("Bias-variance tradeoff for the franke function")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "Bias_var_franke.pdf"))
#plt.show()
