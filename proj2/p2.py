from NN import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns
from functions import *

np.random.seed(123)

n = 100
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()

target = np.ravel(FrankeFunction(x, y))
target = target.reshape(target.shape[0], 1)

poly_degree = 3
X = create_design_matrix(x, y, poly_degree)

X_train, X_test, t_train, t_test = train_test_split(X, target)

input_nodes = X_train.shape[1]
output_nodes = 1

linear_regression = FFNN((input_nodes, output_nodes), output_func=identity, cost_func=CostOLS, seed=123)

linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

scheduler = Constant(eta=1e-2)
scores = linear_regression.fit(X_train, t_train, scheduler, lam=1e-4, epochs=2000)
