import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def FrankeFunction(x, y):
    return (
        0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2)) +
        0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1)) +
        0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2)) -
        0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    )

def create_X(x, y, n):
    if len(x.shape) > 1:
        x, y = x.ravel(), y.ravel()
    N, l = len(x), int((n + 1) * (n + 2) / 2)
    X = np.ones((N, l))
    for i in range(1, n + 1):
        q = int(i * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X

def MSE(y_actual, y_model):
    return np.mean((y_actual - y_model)**2)

def R2_score(y_actual, y_model):
    return 1 - np.sum((y_actual - y_model)**2) / np.sum((y_actual - np.mean(y_actual))**2)

N = 100
maxdegree = 5
max_noise = 1

np.random.seed(1)
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
noise = np.random.normal(0, max_noise, x.shape)
z = FrankeFunction(x, y) + noise

scaler = StandardScaler()
degrees = np.arange(1, maxdegree+1, 1)
train_mse, test_mse, train_r2, test_r2, beta_values = [], [], [], [], []

for degree in degrees:
    X = create_X(x, y, n=degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    scaler.fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train
    z_tilde, z_predict = X_train_scaled @ beta, X_test_scaled @ beta
    beta_values.append(beta)
    train_mse.append(MSE(z_train, z_tilde))
    test_mse.append(MSE(z_test, z_predict))
    train_r2.append(R2_score(z_train, z_tilde))
    test_r2.append(R2_score(z_test, z_predict))

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot MSE on the first subplot
ax1.plot(degrees, train_mse,".--", label="Train")
ax1.plot(degrees, test_mse,".-", label="Test")
ax1.legend()
ax1.set_title("Mean squared error")
ax1.set_xlabel("Polynomial degree")
ax1.set_ylabel("MSE")

# Plot R2 scores on the second subplot
ax2.plot(degrees, train_r2, ".--", label="Train")
ax2.plot(degrees, test_r2, ".-", label="Test")
ax2.legend()
ax2.set_title("R2 score")
ax2.set_xlabel("Polynomial degree")
ax2.set_ylabel("$R^2$")

plt.tight_layout()
plt.show()

# Plot beta parameters
for degree, beta in zip(degrees, beta_values):
    plt.bar(range(beta.size), beta, label=f"{degree=}")
plt.legend()
plt.xticks(np.arange(plt.xlim()[1], step=2))
plt.title("Beta parameters")
plt.xlabel("$i$")
plt.ylabel(r"$\beta_i$")
plt.show()
