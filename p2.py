import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the Franke function with added noise
def franke_function(x, y, noise_std=0.1):
    term1 = 0.75 * np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4)
    term2 = 0.75 * np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10)
    term3 = 0.5 * np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4)
    term4 = 0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 - term4 + noise_std * np.random.randn(*x.shape)

# Generate random data points in the range [0, 1]
np.random.seed(0)
n_points = 1000
x = np.random.rand(n_points)
y = np.random.rand(n_points)
z = franke_function(x, y, noise_std=0.1)

# Split the data into training and test sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.33, random_state=42)

# Polynomial regression up to degree 5
max_degree = 5
mse_scores_train = []
mse_scores_test = []
r2_scores_train = []
r2_scores_test = []
beta_values = []

# Ridge regression parameter (lambda)
lambda_values = [0.001, 0.01, 0.1, 1, 10]

for lambda_val in lambda_values:
    mse_train_lambda = []
    mse_test_lambda = []
    r2_train_lambda = []
    r2_test_lambda = []
    beta_lambda = []

    for degree in range(1, max_degree+1):
        X_train = np.column_stack([x_train**i * y_train**(degree-i) for i in range(degree+1)])
        X_test = np.column_stack([x_test**i * y_test**(degree-i) for i in range(degree+1)])

        # Center and scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform Ridge regression
        I = np.identity(X_train_scaled.shape[1])
        beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled + lambda_val * I) @ X_train_scaled.T @ z_train

        # Make predictions on training data
        z_pred_train = X_train_scaled @ beta

        # Make predictions on test data
        z_pred_test = X_test_scaled @ beta

        # Calculate MSE and R-squared for training and test data
        mse_train = mean_squared_error(z_train, z_pred_train)
        mse_test = mean_squared_error(z_test, z_pred_test)
        r2_train = r2_score(z_train, z_pred_train)
        r2_test = r2_score(z_test, z_pred_test)

        mse_train_lambda.append(mse_train)
        mse_test_lambda.append(mse_test)
        r2_train_lambda.append(r2_train)
        r2_test_lambda.append(r2_test)
        beta_lambda.append(beta)

    mse_scores_train.append(mse_train_lambda)
    mse_scores_test.append(mse_test_lambda)
    r2_scores_train.append(r2_train_lambda)
    r2_scores_test.append(r2_test_lambda)
    beta_values.append(beta_lambda)

# Plot MSE and R-squared scores for different lambda values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, lambda_val in enumerate(lambda_values):
    plt.plot(range(1, max_degree+1), mse_scores_train[i], ".--", label=f"Training, λ={lambda_val}")
    plt.plot(range(1, max_degree+1), mse_scores_test[i], ".-", label=f"Test, λ={lambda_val}")
plt.title("Mean Squared Error (MSE) with Ridge Regression")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.legend()

plt.subplot(1, 2, 2)
for i, lambda_val in enumerate(lambda_values):
    plt.plot(range(1, max_degree+1), r2_scores_train[i], ".--", label=f"Training, λ={lambda_val}")
    plt.plot(range(1, max_degree+1), r2_scores_test[i], ".-", label=f"Test, λ={lambda_val}")
plt.title("R-squared (R^2) Score with Ridge Regression")
plt.xlabel("Polynomial Degree")
plt.ylabel("R^2")
plt.legend()

plt.tight_layout()
plt.show()

# Plot beta values for a specific lambda (e.g., lambda=0.01)
selected_lambda_idx = 1  # Change the index to select a different lambda value
selected_lambda = lambda_values[selected_lambda_idx]
plt.figure(figsize=(12, 6))
for i in range(max_degree):
    plt.plot(range(i+2), beta_values[selected_lambda_idx][i], ".-", label=f"Degree {i+1}")
plt.title(f"Estimated Beta Coefficients with Ridge Regression (λ={selected_lambda})")
plt.xlabel("Beta Index")
plt.ylabel("Beta Value")
plt.legend()
plt.show()
