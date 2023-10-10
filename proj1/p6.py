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

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Custom Ridge regression function
def Ridge_fit_beta(X, y, alpha):
    I = np.eye(X.shape[1])
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

def create_design_matrix(x, y, degree):
    if len(x.shape) > 1:
        x, y = x.ravel(), y.ravel()
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)
    X = np.ones((N, l))
    #col = 1
    for i in range(1, degree + 1):
        col = int((i)*(i+1)/2)
        for j in range(i + 1):
            X[:, col+j] = x ** (i - j) * y ** j
            #col += 1
    return X


def k_fold(data, k):
    N = len(data)
    fold_size = N // k
    ind = np.arange(N)
    np.random.shuffle(ind)
    k_fold_ind = []
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_ind = ind[test_start:test_end]
        train_ind = np.concatenate([ind[:test_start], ind[test_end:]])
        k_fold_ind.append((train_ind, test_ind))
    return k_fold_ind


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
data = np.column_stack((x, y))
#x = np.random.rand(n)
#y = 3*x**2 + np.random.randn(n)

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n * n)
z = z + noise
z.ravel()

#data = np.column_stack((x, z))

# Lambda values
nlambdas = 6
lambdas = np.logspace(-3, 3, nlambdas)

degrees = np.arange(1, max_degree+1)
k = 5

scaler = StandardScaler()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

#plt.figure(figsize=(10, 6))
# Ridge Regression
for i, lmb in enumerate(lambdas):
    mse_per_degree_ridge = []
    for degree in degrees:
        scores_KFold = np.zeros(k)
        k_fold_indices = k_fold(data, k)
        for j, (train_indices, test_indices) in enumerate(k_fold_indices):
            xtrain, ztrain = data[train_indices, 0], z[train_indices]
            xtest, ztest = data[test_indices, 0], z[test_indices]
            Xtrain = create_design_matrix(xtrain, ztrain, degree=degree)
            ridge = Ridge(alpha=lmb)
            ridge.fit(Xtrain, ztrain)
            Xtest = create_design_matrix(xtest, ztest, degree=degree)
            scaler.fit(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
            zpred = ridge.predict(Xtest_scaled)
            scores_KFold[j] = MSE(ztest, zpred)
        mse_per_degree_ridge.append(np.mean(scores_KFold))
    ax1.plot(degrees, mse_per_degree_ridge,'.-', label=f'Ridge, λ={lmb:.1e}')

ax1.set_xlabel('Degree')
ax1.set_ylabel('MSE (log scale)')
ax1.set_yscale('log')
ax1.legend()
ax1.grid()
#plt.show()



# Lasso Regression
#plt.figure(figsize=(10, 6))
#for i, lmb in enumerate(lambdas):
mse_per_degree_lasso = []
mse_per_degree_lasso_train = []
for degree in degrees:
    scores_KFold = np.zeros(k)
    scores_KFold_train = np.zeros(k)
    k_fold_indices = k_fold(data, k)
    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        xtrain, ztrain = data[train_indices, 0], z[train_indices]
        xtest, ztest = data[test_indices, 0], z[test_indices]
        Xtrain = create_design_matrix(xtrain, ztrain, degree=degree)
        scaler.fit(Xtrain)
        Xtrain_scaled = scaler.transform(Xtrain)
        lasso = Lasso(alpha=lmb, max_iter=10000)
        lasso.fit(Xtrain_scaled, ztrain)
        Xtest = create_design_matrix(xtest, ztest, degree=degree)
        scaler.fit(Xtest)
        Xtest_scaled = scaler.transform(Xtest)
        zpred_test = lasso.predict(Xtest_scaled)
        zpred_train = lasso.predict(Xtrain_scaled)
        scores_KFold[j] = mean_squared_error(ztest, zpred_test)
        scores_KFold_train[j] = mean_squared_error(ztrain, zpred_train)
    mse_per_degree_lasso.append(np.mean(scores_KFold))
    mse_per_degree_lasso_train.append(np.mean(scores_KFold_train))
ax2.plot(degrees, mse_per_degree_lasso,'.-', label=f'Lasso, test')
ax2.plot(degrees, mse_per_degree_lasso_train,'.-', label=f'Lasso, train')

ax2.set_xlabel('Degree')
ax2.set_ylabel('MSE (log scale)')
ax2.set_yscale('log')
ax2.grid()
ax2.legend()
#plt.show()




# OLS Regression
mse_per_degree_OLS = []
mse_per_degree_OLS_train = []
for degree in degrees:
    scores_KFold = np.zeros(k)
    train_scores_KFold = np.zeros(k)
    k_fold_indices = k_fold(data, k)
    for j, (train_indices, test_indices) in enumerate(k_fold_indices):
        xtrain, ztrain = x[train_indices], z[train_indices]
        xtest, ztest = x[test_indices], z[test_indices]
        Xtrain = create_design_matrix(xtrain, ztrain, degree=degree)
        scaler.fit(Xtrain)
        Xtrain_scaled = scaler.transform(Xtrain)
        beta = OLS_fit_beta(Xtrain_scaled, ztrain)

        Xtest = create_design_matrix(xtest, ztest, degree=degree)
        scaler.fit(Xtest)
        Xtest_scaled = scaler.transform(Xtest)
        #beta = OLS_fit_beta(Xtest_scaled, ztrain)

        ztilde, zpred = Xtrain_scaled @ beta, Xtest_scaled @ beta
        scores_KFold[j] = MSE(ztest, zpred) #np.mean((zpred.ravel() - ztest) ** 2)
        train_scores_KFold[j] = MSE(ztrain, ztilde)
    mse_per_degree_OLS.append(np.mean(scores_KFold))
    mse_per_degree_OLS_train.append(np.mean(train_scores_KFold))
ax3.plot(degrees, mse_per_degree_OLS,'.-', label=f'OLS, test')
ax3.plot(degrees, mse_per_degree_OLS_train,'.-', label=f'OLS, train')

ax3.set_xlabel('Degree')
ax3.set_ylabel('MSE (log scale)')
ax3.set_yscale('log')
ax3.legend()
ax3.grid()
plt.show()
