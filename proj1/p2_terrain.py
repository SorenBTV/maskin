import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from imageio import imread

# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

# Define a function to fit beta coefficients using Ordinary Least Squares (OLS)
def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Define a function to calculate R-squared (R2) score
def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()
    return 1 - np.sum((y_actual - y_model) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)

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


# Load terrain data from an image file
terrain = imread('SRTM_data_Norway_1.tif')
n = 100
terrain = terrain[:n, :n]

# Create a mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()

z = terrain.ravel()
mean_scale = np.mean(z)
std_scale = np.std(z)
z = (z - mean_scale) / std_scale

# Initialize a StandardScaler
scaler = StandardScaler()

# Set polynomial degree and a range of lambda values
degrees = np.array([6])
lambda_values = np.logspace(-5, 0, 6)
test_mse = np.zeros(len(lambda_values))
train_mse = np.zeros(len(lambda_values))

# Loop through lambda values
for i in range(len(lambda_values)):

    # Create a design matrix for polynomial regression
    X = create_design_matrix(x, y, degrees[0])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scale the feature matrix using StandardScaler
    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

    # Create and fit Ridge regression model using custom function
    beta = Ridge_fit_beta(X_train_scaled, z_train, lambda_values[i])

    # Calculate predictions on training and testing data
    z_train_pred = X_train_scaled @ beta
    z_test_pred = X_test_scaled @ beta

    # Calculate and store Mean Squared Error for training and testing data
    test_mse[i] =MSE(z_test, z_test_pred)
    train_mse[i] =MSE(z_train, z_train_pred)

# Plot MSE for different lambda values
plt.plot(lambda_values,train_mse, ".--", label="train")
plt.plot(lambda_values,test_mse, ".-", label="test")
plt.title("MSE for terrain data using ridge regression")
plt.xscale("log")
plt.xlabel(r'$\lambda$')
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.savefig("figures\Ridge_terrain.pdf", dpi=300)
#plt.show()
