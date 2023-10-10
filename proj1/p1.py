import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as R2_score


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def R2_score(y_actual, y_model):
    #Returns the R2 score of the two arrays.
    y_actual, y_model = y_actual.ravel(), y_model.ravel()  # flatten arrays
    return 1 - np.sum((y_actual - y_model)**2) / np.sum((y_actual - np.mean(y_actual))**2)


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

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


np.random.seed(123)  # Setting a seed for reproducibility
n = 25  # Number of data points

#Generate random values for x and y within [0, 1]
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()
#print(x)

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n*n)
z = z + noise


# Create polynomial features up to fifth order
max_degree = 14
#poly = PolynomialFeatures(degree=5)

scaler = StandardScaler()
degrees = np.arange(0, max_degree, 1)
train_mse = np.empty(degrees.shape)
test_mse = np.empty_like(train_mse)
train_r2 = np.empty_like(train_mse)
test_r2 = np.empty_like(train_mse)
beta_values = []

for degree in degrees:
    X = create_design_matrix(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)


    beta = OLS_fit_beta(X_train_scaled, z_train)
    z_tilde, z_predict = X_train_scaled @ beta, X_test_scaled @ beta
    beta_values.append(beta)
    train_mse[degree] = (MSE(z_train, z_tilde))
    test_mse[degree] = (MSE(z_test, z_predict))
    train_r2[degree] = (R2_score(z_train, z_tilde))
    test_r2[degree] = (R2_score(z_test, z_predict))


# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot MSE on the first subplot
ax1.plot(degrees, train_mse,".--", label="Train")
ax1.plot(degrees, test_mse,".-", label="Test")
ax1.legend()
ax1.grid()
ax1.set_title("Mean squared error")
ax1.set_xlabel("Polynomial degree")
ax1.set_ylabel("MSE")

# Plot R2 scores on the second subplot
ax2.plot(degrees, train_r2, ".--", label="Train")
ax2.plot(degrees, test_r2, ".-", label="Test")
ax2.legend()
ax2.grid()
ax2.set_title("R2 score")
ax2.set_xlabel("Polynomial degree")
ax2.set_ylabel("$R^2$")

plt.tight_layout()
plt.show()

"""
#Beta plot
for degree, beta in zip(degrees, beta_values):
    plt.plot(range(beta.size), beta, label=f"{degree=}")
# Customize the x-axis ticks and tilt them by 60 degrees
x_ticks = np.arange(beta_values[0].size)
plt.xticks(x_ticks, labels=x_ticks, rotation=60)

plt.legend()
plt.xticks(np.arange(plt.xlim()[1]), labels=np.arange(plt.xlim()[1]))
plt.title("Beta parameters")
plt.xlabel("$i$")
plt.ylabel(r"$\beta_i$")
plt.show()
"""
