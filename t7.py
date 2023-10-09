import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

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

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

# Initialize variables
np.random.seed(123)  # Setting a seed for reproducibility
n = 100  # Number of data points
max_degree = 15  # Maximum polynomial degree

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n * n)
z = z + noise

degrees = np.arange(0, max_degree, 1)
error_train_ols = np.zeros(max_degree)
error_test_ols = np.zeros(max_degree)
error_train_ridge = np.zeros(max_degree)
error_test_ridge = np.zeros(max_degree)
error_train_lasso = np.zeros(max_degree)
error_test_lasso = np.zeros(max_degree)

# Adding OLS, Ridge, and Lasso regression
kf = KFold(n_splits=5, shuffle=True, random_state=123)

bias_ols = np.zeros(max_degree)
variance_ols = np.zeros(max_degree)
bias_ridge = np.zeros(max_degree)
variance_ridge = np.zeros(max_degree)
bias_lasso = np.zeros(max_degree)
variance_lasso = np.zeros(max_degree)

for degree in degrees:
    X = create_design_matrix(x, z.flatten(), degree)
    mse_train_ols = np.zeros(max_degree)
    mse_test_ols = np.zeros(max_degree)
    mse_train_ridge = np.zeros(max_degree)
    mse_test_ridge = np.zeros(max_degree)
    mse_train_lasso = np.zeros(max_degree)
    mse_test_lasso = np.zeros(max_degree)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]

        # OLS regression
        ols_model = LinearRegression()
        ols_model.fit(X_train, z_train)
        z_predict_ols = ols_model.predict(X_test)
        mse_train_ols[degree] = (MSE(z_train, ols_model.predict(X_train)))
        mse_test_ols[degree] = (MSE(z_test, z_predict_ols))

        # Ridge regression
        alpha_ridge = 0.01  # You can adjust the alpha (regularization strength)
        ridge_model = Ridge(alpha=alpha_ridge)
        ridge_model.fit(X_train, z_train)
        z_predict_ridge = ridge_model.predict(X_test)
        mse_train_ridge[degree] = (MSE(z_train, ridge_model.predict(X_train)))
        mse_test_ridge[degree] = (MSE(z_test, z_predict_ridge))

        # Lasso regression
        alpha_lasso = 0.01  # You can adjust the alpha (regularization strength)
        lasso_model = Lasso(alpha=alpha_lasso)
        lasso_model.fit(X_train, z_train)
        z_predict_lasso = lasso_model.predict(X_test)
        mse_train_lasso[degree] = (MSE(z_train, lasso_model.predict(X_train)))
        mse_test_lasso[degree] = (MSE(z_test, z_predict_lasso))

    error_train_ols[degree] = np.mean(mse_train_ols)
    error_test_ols[degree] = np.mean(mse_test_ols)
    error_train_ridge[degree] = np.mean(mse_train_ridge)
    error_test_ridge[degree] = np.mean(mse_test_ridge)
    error_train_lasso[degree] = np.mean(mse_train_lasso)
    error_test_lasso[degree] = np.mean(mse_test_lasso)

    # Calculate bias and variance for OLS
    bias_ols[degree] = np.mean((z_test - np.mean(z_predict_ols)) ** 2)
    variance_ols[degree] = np.mean(np.var(z_predict_ols))

    # Calculate bias and variance for Ridge
    bias_ridge[degree] = np.mean((z_test - np.mean(z_predict_ridge)) ** 2)
    variance_ridge[degree] = np.mean(np.var(z_predict_ridge))

    # Calculate bias and variance for Lasso
    bias_lasso[degree] = np.mean((z_test - np.mean(z_predict_lasso)) ** 2)
    variance_lasso[degree] = np.mean(np.var(z_predict_lasso))

# Plotting Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(degrees, error_train_ols, ".--", label="OLS Train Error")
plt.plot(degrees, error_test_ols, label="OLS Test Error")
plt.plot(degrees, error_train_ridge, ".--", label="Ridge Train Error")
plt.plot(degrees, error_test_ridge, label="Ridge Test Error")
plt.plot(degrees, error_train_lasso, ".--", label="Lasso Train Error")
plt.plot(degrees, error_test_lasso, label="Lasso Test Error")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.title("MSE for OLS, Ridge, and Lasso Regression")

plt.subplot(1, 2, 2)
plt.plot(degrees, error_test_ols, ".--", label="OLS Test Error")
plt.plot(degrees, bias_ols, label="Bias (OLS)")
plt.plot(degrees, variance_ols, label="Variance (OLS)")
plt.plot(degrees, bias_ridge, label="Bias (Ridge)")
plt.plot(degrees, variance_ridge, label="Variance (Ridge)")
plt.plot(degrees, bias_lasso, label="Bias (Lasso)")
plt.plot(degrees, variance_lasso, label="Variance (Lasso)")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.title("Test Error, Bias, and Variance for OLS, Ridge, and Lasso")

plt.tight_layout()
plt.show()
