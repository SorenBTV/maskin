import autograd.numpy as np
from autograd import elementwise_grad


def CostOLS(target):

    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):

    def func(X):

        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target, weights={0: 1.0, 1: 3.0}):
    """
    Computes the weighted cross-entropy loss function.

    Parameters:
    - target: Ground truth labels (0 or 1).
    - weights: Dictionary specifying weights for Class 0 and Class 1.

    Returns:
    - A function that computes the weighted cross-entropy loss for predicted probabilities.
    """
    def func(X):
        # Predictions are probabilities for Class 1
        return -(1.0 / target.size) * np.sum(
            weights[1] * target * np.log(X + 1e-10) +
            weights[0] * (1 - target) * np.log(1 - X + 1e-10)
        )
    return func



def identity(X):
    return X

def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)
