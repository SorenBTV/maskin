import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the Franke function with added noise
def franke_function(x,y, noise_std=1):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noise_std* np.random.normal(0, 1, x.shape)


# Generate random data points in the range [0, 1]
np.random.seed(0)
n_points = 100
x = np.sort(np.random.uniform(0, 1, n_points))
y = np.sort(np.random.uniform(0, 1, n_points))
#x,y = np.meshgrid(x, y)
z = franke_function(x, y, noise_std=1)


# Split the data into training and test sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)#, random_state=42)

# Polynomial regression up to degree 5
max_degree = 5
mse_scores_train = []
mse_scores_test = []
r2_scores_train = []
r2_scores_test = []
beta_values = []

# Initialize scaler
scaler = StandardScaler()



for degree in range(1, max_degree+1):
    X_train = np.column_stack([x_train**i * y_train**(degree-i) for i in range(degree+1)])
    X_test = np.column_stack([x_test**i * y_test**(degree-i) for i in range(degree+1)])

    # Center and scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform ordinary least squares regression
    beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

    # Make predictions on training data
    z_pred_train = X_train_scaled @ beta

    # Make predictions on test data
    z_pred_test = X_test_scaled @ beta

    # Calculate MSE and R-squared for training and test data
    mse_train = mean_squared_error(z_train, z_pred_train)
    mse_test = mean_squared_error(z_test, z_pred_test)
    r2_train = r2_score(z_train, z_pred_train)
    r2_test = r2_score(z_test, z_pred_test)

    mse_scores_train.append(mse_train)
    mse_scores_test.append(mse_test)
    r2_scores_train.append(r2_train)
    r2_scores_test.append(r2_test)
    beta_values.append(beta)


# Plot MSE and R-squared scores for training and test data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, max_degree+1), mse_scores_train, ".--", label="Training")
plt.plot(range(1, max_degree+1), mse_scores_test, ".-", label="Test")
plt.title("Mean Squared Error (MSE)")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, max_degree+1), r2_scores_train, ".--", label="Training")
plt.plot(range(1, max_degree+1), r2_scores_test, ".-", label="Test")
plt.title("R-squared (R^2) Score")
plt.xlabel("Polynomial Degree")
plt.ylabel("R^2")
plt.legend()

plt.tight_layout()
plt.show()

"""
for d in range(max_degree):
    plt.bar(
            range(beta_values[d].size),  # indices
            beta_values[d],
            label=f"{d=}",
    )
plt.legend()
plt.xticks(np.arange(plt.xlim()[1], step=2))
plt.title("Beta parameters")
plt.xlabel("$i$")
plt.ylabel(r"$\beta_i$")
plt.show()


# Plot beta values
plt.figure(figsize=(12, 6))
for i in range(max_degree):
    plt.plot(range(i+2), beta_values[i], ".-", label=f"Degree {i+1}")
plt.title("Estimated Beta Coefficients")
plt.xlabel("Polynomial degree")
plt.ylabel("Beta Value")
plt.legend()
plt.show()
"""
