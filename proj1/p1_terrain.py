import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imageio import imread
import seaborn as sns

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def R2_score(y_actual, y_model):
    y_actual, y_model = y_actual.ravel(), y_model.ravel()  # flatten arrays
    return 1 - np.sum((y_actual - y_model) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)

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

# Load the terrain
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

max_degree = 5

degrees = np.arange(1, max_degree + 1, 1)
scaler = StandardScaler()
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
    beta_values.append(beta)
    z_tilde, z_predict = X_train_scaled @ beta, X_test_scaled @ beta
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
plt.title("MSE for terrain data using OLS")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.savefig("figures\MSE_OLS_terrain.pdf")
#plt.show()

# Plot R2 scores
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_r2, ".--", label="Train")
plt.plot(degrees, test_r2, ".-", label="Test")
plt.legend()
plt.grid()
plt.title("R2 score terrain data")
plt.xlabel("Polynomial degree")
plt.ylabel("$R^2$")
plt.savefig("figures\R2_score_terrain.pdf")
#plt.show()

plt.figure(figsize=(8, 5))
for degree in degrees:
    plt.scatter(
            range(beta_values[degree-1].size),  # indices
            beta_values[degree-1],
            label=f"{degree=}",
    )
plt.legend()
plt.xticks(np.arange(plt.xlim()[1], step=2))
plt.title("Beta parameters terrain data")
plt.xlabel("$i$")
plt.ylabel(r"$\beta_i$")
plt.grid()
plt.savefig("figures\B_terrain.pdf")
#plt.show()
