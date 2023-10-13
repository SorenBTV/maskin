import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()
    return 1 - np.sum((y_actual - y_model) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)

# Custom Ridge regression function
def Ridge_fit_beta(X, y, alpha):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

# Function to create a design matrix
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
n = 50  # Number of data points

#Generate random values for x and y within [0, 1]
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n*n)
z = z + noise

# Create polynomial features up to fifth order
#max_degree = 14

# Initialize a StandardScaler
scaler = StandardScaler()

# Polynomial degrees to consider
#degrees = np.arange(0, max_degree, 1)
degrees = np.array([15])

# Values of lambda (regularization strength) to test
lambda_values = np.logspace(-10, 0, 11)
test_mse = np.zeros(len(lambda_values))
train_mse = np.zeros(len(lambda_values))

for i in range(len(lambda_values)):

    X = create_design_matrix(x, y, degrees[0])
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)#, random_state=42)

    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

    # Create and fit Ridge regression model using custom function
    beta = Ridge_fit_beta(X_train_scaled, z_train, lambda_values[i])

    z_train_pred = X_train_scaled @ beta
    z_test_pred = X_test_scaled @ beta
    test_mse[i] =MSE(z_test, z_test_pred)
    train_mse[i] =MSE(z_train, z_train_pred)

#print(test_mse[0,:])
plt.plot(lambda_values,test_mse, ".-", label="test")
plt.plot(lambda_values,train_mse, ".--", label="train")
plt.title("MSE plotted against different $\lambda$ values")
plt.xscale("log")
plt.xlabel(r'$\lambda$')
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.savefig("figures\Ridge_frankefunction.pdf", dpi=300)
plt.show()
