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

# Set a random seed for reproducibility
np.random.seed(2018)

# Number of data points
n = 20

# Generate random values for x and y within [0, 1]
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x,y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 1, n*n)
z = z + noise

# Maximum polynomial degree
max_degree = 15

# Initialize a StandardScaler
scaler = StandardScaler()

# Polynomial degrees to consider
degrees = np.array([10])
#degrees = np.arange(0, max_degree, 1)

# Values of lambda (regularization strength) to test
lambda_values = [10e-6, 10e-5, 10e-4, 10e-3, 10e-2]
#lambda_values = [0.01, 0.1]

# Store results
results = []
#train_mse
test_mse = np.zeros((max_degree, len(lambda_values)))

for i in range(len(lambda_values)):


    for j in range(len(degrees)):
        X = create_design_matrix(x, y, degrees[j])
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

        scaler.fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        # Create and fit Ridge regression model using custom function
        beta = Ridge_fit_beta(X_train_scaled, z_train, lambda_values[i])

        z_train_pred = X_train_scaled @ beta
        z_test_pred = X_test_scaled @ beta
        test_mse[j][i] =MSE(z_test, z_test_pred)

print(test_mse[0,:])
plt.plot(lambda_values,test_mse[0,:])
plt.show()

"""
plt.contourf(test_mse)
plt.colorbar()
plt.xlabel("$lambda$ (log 10)")
plt.ylabel("Polynomial degree")
plt.show()
"""


"""
# Loop over lambda values
for lambda_value in lambda_values:
    train_mse = []
    test_mse = []
    train_r2 = []
    test_r2 = []

    # Loop over polynomial degrees
    for degree in degrees:
        X = create_design_matrix(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

        scaler.fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        # Create and fit Ridge regression model using custom function
        beta = Ridge_fit_beta(X_train_scaled, z_train, lambda_value)

        z_train_pred = X_train_scaled @ beta
        z_test_pred = X_test_scaled @ beta

        train_mse.append(MSE(z_train, z_train_pred))
        test_mse.append(MSE(z_test, z_test_pred))
        train_r2.append(R2_score(z_train, z_train_pred))
        test_r2.append(R2_score(z_test, z_test_pred))

    # Store results for this lambda value
    results.append({
        "lambda": lambda_value,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# Plot the results (rest of your code remains the same)
plt.figure(figsize=(12, 6))

# Plot MSE
for result in results:
    plt.plot(degrees, result["test_mse"], ".-", label=f"λ={result['lambda']}")
    plt.plot(degrees, result["train_mse"], ".--", label=f"train λ={result['lambda']}")
plt.legend()
plt.title("Mean Squared Error for Ridge regression")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")



# Plot R-squared
plt.subplot(1, 2, 2)
for result in results:
    plt.plot(degrees, result["test_r2"], ".-", label=f"λ={result['lambda']}")
    plt.plot(degrees, result["train_r2"], ".--", label=f"train λ={result['lambda']}")
plt.legend()
plt.title("R-squared Score for Ridge regression")
plt.xlabel("Polynomial Degree")
plt.ylabel("$R^2$")

plt.tight_layout()

plt.show()
"""
