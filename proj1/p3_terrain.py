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

# Load the terrain data from an image file
terrain = imread('SRTM_data_Norway_1.tif')
n = 100
terrain = terrain[:n, :n]

# Creates a mesh of image pixels
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

# Maximum polynomial degree
max_degree = 14

# Polynomial degrees to consider
degrees = np.arange(1, max_degree+1, 1)
train_mse = np.empty(degrees.shape)
test_mse = np.empty_like(train_mse)
train_r2 = np.empty_like(train_mse)
test_r2 = np.empty_like(train_mse)
lasso_alpha = 0.001  # Lasso regularization parameter

# Loop through polynomial degrees
for degree in degrees:

    # Create a design matrix for polynomial regression
    X = create_design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scale the feature matrix using StandardScaler
    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

    # Create and fit Lasso regression model
    lasso = Lasso(alpha=lasso_alpha, max_iter=100000)
    lasso.fit(X_train_scaled, z_train)

    z_train_pred = lasso.predict(X_train_scaled)
    z_test_pred = lasso.predict(X_test_scaled)

    # Calculate and store Mean Squared Error and R-squared (R2) scores
    train_mse[degree-1] = mean_squared_error(z_train, z_train_pred)
    test_mse[degree-1] = mean_squared_error(z_test, z_test_pred)
    train_r2[degree-1] = r2_score(z_train, z_train_pred)
    test_r2[degree-1] = r2_score(z_test, z_test_pred)

# Plot MSE for different polynomial degrees
plt.plot(degrees, train_mse,".--", label="Train")
plt.plot(degrees, test_mse,".-", label="Test")
plt.legend()
plt.grid()
plt.title("MSE for terrain data using lasso regression")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.savefig("figures\MSE_Lasso_terrain.pdf")
#plt.show()
