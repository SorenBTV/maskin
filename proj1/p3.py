import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

# Definition of the Franke Function
def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

np.random.seed(123)  # Setting a seed for reproducibility
n = 100  # Number of data points

#Generate random values for x and y within [0, 1]
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n*n)
z = z + noise

# Maximum polynomial degree
max_degree = 14

# Initialize a StandardScaler
scaler = StandardScaler()

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

    # Calculate and store Mean Squared Error (MSE)
    train_mse[degree-1] = mean_squared_error(z_train, z_train_pred)
    test_mse[degree-1] = mean_squared_error(z_test, z_test_pred)
    train_r2[degree-1] = r2_score(z_train, z_train_pred)
    test_r2[degree-1] = r2_score(z_test, z_test_pred)

# Plot MSE for different polynomial degrees
plt.plot(degrees, train_mse,".--", label="Train")
plt.plot(degrees, test_mse,".-", label="Test")
plt.legend()
plt.grid()
plt.title("MSE for Franke function using lasso regression")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.savefig("figures\MSE_Lasso_franke.pdf")
#plt.show()
