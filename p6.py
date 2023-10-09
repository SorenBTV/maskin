import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Ridge_fit_beta(X, y,L,d):
    I = np.eye(d,d)
    return np.linalg.pinv(X.T @ X + L*I) @ X.T @ y


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
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

"""
def ridge(x, y, z, nlambdas, degree)
    i = 0
    for lmb in lambdas:
        ridge = Ridge(alpha = lmb)
        j = 0

        X = create_design_matrix(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z.flatten(), test_size=0.2)

        Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        ridge.fit(Xtrain, ztrain[:, np.newaxis])

        Xtest = poly.fit_transform(xtest[:, np.newaxis])
        zpred = ridge.predict(Xtest)

        scores_KFold[i,j] = np.sum((zpred - ztest[:, np.newaxis])**2)/np.size(zpred)

        j += 1
    i += 1

    estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

    return estimated_mse_KFold
"""

# Initialize variables
np.random.seed(123)  # Setting a seed for reproducibility
n = 100  # Number of data points
n_bootstraps = 100  # Number of bootstrap samples
max_degree = 10  # Maximum polynomial degree
Lambda = 0.01

x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
x, y = x.ravel(), y.ravel()

# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)
noise = np.random.normal(0, 0.1, n * n)
z = z + noise

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = max_degree)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-10, 0, nlambdas)


#Split data, no scaling is used and we include the intercept

degrees = np.arange(0, max_degree, 1)

for degree in degrees:
    X = create_design_matrix(x, y, max_degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    beta_OLS = OLS_fit_beta(X_train, z_train)
    beta_Ridge = Ridge_fit_beta(X_train, z_train,Lambda,max_degree)
    print(beta_OLS)
    print(beta_Ridge)
    #predict value
    ztilde_test_OLS = X_test @ beta_OLS
    ztilde_test_Ridge = X_test @ beta_Ridge

    #Calculate MSE
    print("  ")
    print("test MSE of OLS:")
    print(MSE(z_test,ztilde_test_OLS))
    print("  ")
    print("test MSE of Ridge")
    print(MSE(z_test,ztilde_test_Ridge))

plt.scatter(x,z,label='Data')
plt.plot(x, X @ beta_OLS,'*', label="OLS_Fit")
plt.plot(x, X @ beta_Ridge, label="Ridge_Fit")
plt.grid()
plt.legend()
plt.show()




"""
# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))
"""

"""
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(x):
        xtrain = x[train_inds]
        ztrain = z[train_inds]

        xtest = x[test_inds]
        ztest = z[test_inds]

        Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        ridge.fit(Xtrain, ztrain[:, np.newaxis])

        Xtest = poly.fit_transform(xtest[:, np.newaxis])
        zpred = ridge.predict(Xtest)

        scores_KFold[i,j] = np.sum((zpred - ztest[:, np.newaxis])**2)/np.size(zpred)

        j += 1
    i += 1

estimated_mse_KFold = np.mean(scores_KFold, axis = 1)





## Plot and compare the slightly different ways to perform cross-validation

plt.figure()

#plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()
"""
