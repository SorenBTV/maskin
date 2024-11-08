import os
import numpy as np
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from cost import *
from scheduler import *
from functions import *
from sklearn.datasets import load_wine



# Multi-class logistic regression with SGD
def logistic_regression_sgd(X, y, scheduler=None, learning_rate=0.001, epochs=100, batch_size=8, lambda_val=1e-5, mom=0.0, seed=123):
    n_samples, n_features = X.shape
    n_classes = y.shape[1]  # Number of classes after one-hot encoding
    weights = np.zeros((n_features, n_classes))  # Initialize weights for each class
    cost_gradient = grad(lambda w, y, X: CostLogReg(y)(softmax(X @ w)))  # Gradient of multi-class logistic cost

    #scheduler initialization
    if scheduler=="ada":
        scheduler = Adagrad(learning_rate)
    elif scheduler=="rms":
        scheduler = RMS_prop(learning_rate, rho=0.9)
    elif scheduler=="adam":
        scheduler = Adam(learning_rate, rho=0.9, rho2=0.999)

    # SGD loop
    for epoch in range(epochs):
        np.random.seed(seed + epoch)
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            gradients = cost_gradient(weights, y_batch, X_batch) + lambda_val * weights * mom
            #Use scheduler to update gradient.
            if scheduler is not None:
                weights -= scheduler.update_change(gradients)
            else:
                weights -= learning_rate * gradients

    return weights


def train(X_train, t_train, scheduler=None, learning_rate=0.001, epochs=100, batch_size=8, lambda_val=1e-5):
    # Train logistic regression model
    weights = logistic_regression_sgd(X_train, t_train, scheduler=scheduler, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, lambda_val=lambda_val)

    # Test model
    t_pred = predict(X_test, weights)
    t_test_labels = np.argmax(t_test, axis=1)  # Convert one-hot encoded t_test to labels for accuracy
    accuracy = np.mean(t_pred == t_test_labels)

    return t_pred, t_test_labels


# Predict function for multi-class
def predict(X, weights):
    logits = X @ weights
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)  # Return class with highest probability



def heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler=None, epochs=100, batch_size=8, mom=0.0, title=None, savename=None):
    accuracy_matrix = np.zeros((len(eta_values), len(lambda_values)))

    for i, eta in enumerate(eta_values):
        for j, lam in enumerate(lambda_values):
            t_pred, t_test_labels = train(X_train, t_train, scheduler=scheduler, learning_rate=eta, epochs=epochs, batch_size=batch_size, lambda_val=lam)

            weights = logistic_regression_sgd(X_train, t_train, learning_rate=eta, lambda_val=lam, mom=mom)
            t_pred = predict(X_test, weights)
            accuracy = np.mean(t_pred == t_test_labels)
            accuracy_matrix[i, j] = accuracy
            #print(f"eta: {eta}, lambda: {lam}, accuracy: {accuracy:.4f}")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=np.log10(lambda_values), yticklabels=eta_values, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 12}, vmin=0.8, vmax=1.0)
    plt.xlabel("Lambda", fontsize=14)
    plt.ylabel("Learning Rate (Eta)", fontsize=14)
    plt.title(f"Accuracy of Logistic Regression using {title}", fontsize=16)

    save_dir = "figs"
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    #plt.show()

    return accuracy_matrix




np.random.seed(123)
seed(123)

data = load_wine()
X, y = data.data, data.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





eta_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]
lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]


heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler=None, epochs=100, batch_size=10, title="no scheduler", savename="log_reg_class_heatmap_eta_lambda_none.pdf")
heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler=None, epochs=100, batch_size=10, mom=0.9, title="momentum", savename="log_reg_class_heatmap_eta_lambda_mom.pdf")
heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler="adam", epochs=100, batch_size=10, title="Adam", savename="log_reg_class_heatmap_eta_lambda_adam.pdf")
heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler="ada", epochs=100, batch_size=10, title="Adagrad", savename="log_reg_class_heatmap_eta_lambda_ada.pdf")
heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler="rms", epochs=100, batch_size=10, title="RMSprop", savename="log_reg_class_heatmap_eta_lambda_rms.pdf")
