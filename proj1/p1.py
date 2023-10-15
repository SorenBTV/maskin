import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define a function to calculate Mean Squared Error (MSE)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Define a function to fit beta coefficients using Ordinary Least Squares (OLS)
def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Define a function to calculate R-squared (R2) score
def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()  # flatten arrays
    return 1 - np.sum((y_actual - y_model)**2) / np.sum((y_actual - np.mean(y_actual))**2)

# Define a function to create a design matrix for polynomial regression
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

# Define the Franke function that generates synthetic data
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


# Set a random seed for reproducibility
np.random.seed(123)

# Define the number of data points and generate random x and y values
n = 100
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()
#print(x)

# Generate synthetic data with noise using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n*n)
z = z + noise


# Specify the maximum polynomial degree and initialize data arrays
max_degree = 5
scaler = StandardScaler()
degrees = np.arange(1, max_degree+1, 1)
train_mse = np.empty(degrees.shape)
test_mse = np.empty_like(train_mse)
train_r2 = np.empty_like(train_mse)
test_r2 = np.empty_like(train_mse)
beta_values = []

# Loop through different polynomial degrees for regression
for degree in degrees:
    X = create_design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)


    beta = OLS_fit_beta(X_train_scaled, z_train)
    z_tilde, z_predict = X_train_scaled @ beta, X_test_scaled @ beta
    beta_values.append(beta)
    train_mse[degree-1] = (MSE(z_train, z_tilde))
    test_mse[degree-1] = (MSE(z_test, z_predict))
    train_r2[degree-1] = (R2_score(z_train, z_tilde))
    test_r2[degree-1] = (R2_score(z_test, z_predict))



# Plot MSE
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_mse,".--", label="Train")
plt.plot(degrees, test_mse,".-", label="Test")
plt.legend()
plt.grid()
plt.title("MSE for Franke function using OLS")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.savefig("figures\MSE_OLS_franke.pdf")
#plt.show()


# Plot R2 scores
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_r2, ".--", label="Train")
plt.plot(degrees, test_r2, ".-", label="Test")
plt.legend()
plt.grid()
plt.title("R2 score Franke function")
plt.xlabel("Polynomial degree")
plt.ylabel("$R^2$")
plt.savefig("figures\R2_score_franke.pdf")
#plt.show()

# Plot beta parameters
plt.figure(figsize=(8, 5))
for degree in degrees:
    plt.scatter(
            range(beta_values[degree-1].size),  # indices
            beta_values[degree-1],
            label=f"{degree=}",
    )
plt.legend()
plt.xticks(np.arange(plt.xlim()[1], step=2))
plt.title("Beta parameters Franke function")
plt.xlabel("$i$")
plt.ylabel(r"$\beta_i$")
plt.grid()
plt.savefig("figures\B_franke.pdf")
#plt.show()
