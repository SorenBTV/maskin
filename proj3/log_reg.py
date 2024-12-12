import os
import numpy as np
import pandas as pd
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.compose import ColumnTransformer
from cost import *
from scheduler import *
from scipy.io import arff
import warnings

# Custom warning handler
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if "Precision is ill-defined" in str(message):
        print()
        print("One class has no predicted samples. Precision set to 0.")
    else:
        warnings.showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = custom_warning_handler


def preprocess_dataset(file_path):
    """
    Description:
    ------------
        Preprocesses the input dataset by encoding
        categorical variables, scaling numerical features,
        and splitting the data into training and testing sets.

    Parameters:
    ------------
        I   file_path (str): Path to the ARFF file containing the dataset.

    Returns:
    ------------
        I   X_train (np.ndarray): Training features.
        II  X_test (np.ndarray): Testing features.
        III y_train (np.ndarray): Training labels.
        IV  y_test (np.ndarray): Testing labels.
    """
    # Loading and converting the ARFF data to a pandas DataFrame
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings in categorical columns
    categorical_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
    for col in categorical_cols:
        df[col] = df[col].str.decode('utf-8')

    # Encode the target variable ('class') into integers
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    # One-hot encode other categorical variables
    df = pd.get_dummies(df, columns=['seismic', 'seismoacoustic', 'shift', 'ghazard'], drop_first=True)

    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']

    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    return X_train, X_test, y_train, y_test


def logistic_regression_sgd(X, y, scheduler=None, learning_rate=0.001, epochs=100, batch_size=4, lambda_val=1e-5, mom=0.0, seed=123):
    """
    Description:
    ------------
        Trains a logistic regression model using stochastic gradient descent (SGD).

    Parameters:
    ------------
        I   X (np.ndarray): Design matrix with n rows (samples) and p columns (features).
        II  y (np.ndarray): Target vector with binary labels (0 and 1).
        III scheduler (str/class): Optimizer used for training (e.g., Adam, Adagrad, RMSprop).
        IV  learning_rate (float): Learning rate for gradient updates.
        V   epochs (int): Number of epochs for training.
        VI  batch_size (int): Size of the batches used in training.
        VII lambda_val (float): Regularization parameter.
        VIII mom (float): Momentum term (for momentum-based optimizers).
        IX  seed (int): Random seed for reproducibility.

    Returns:
    ------------
        I   weights (np.ndarray): Optimized weight vector for logistic regression.
    """

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    velocity = np.zeros_like(weights)

    cost_gradient = grad(lambda w, y, X: CostLogReg(y)(sigmoid(X @ w)))

    if scheduler == "ada":
        scheduler = Adagrad(learning_rate)
    elif scheduler == "rms":
        scheduler = RMS_prop(learning_rate, rho=0.9)
    elif scheduler == "adam":
        scheduler = Adam(learning_rate, rho=0.9, rho2=0.999)

    for epoch in range(epochs):
        np.random.seed(seed + epoch)
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            gradients = cost_gradient(weights, y_batch, X_batch) + lambda_val * weights
            if scheduler is not None:
                weights -= scheduler.update_change(gradients)
            else:
                velocity = mom * velocity - learning_rate * gradients
                weights += velocity
    return weights



def train(X_train, t_train, scheduler=None, learning_rate=0.001, epochs=100, batch_size=8, lambda_val=1e-5, momentum=0.0):
    """
    Description:
    ------------
        Wrapper function to train a logistic regression model using SGD.
        Calls logistic_regression_sgd to optimize weights and evaluates
        the model's performance on the training set.

    Parameters:
    ------------
        I   X_train, t_train (np.ndarray): Training features and labels.
        II  scheduler (str/class): Optimizer used for training (e.g., Adam, Adagrad, RMSprop).
        III learning_rate (float): Learning rate for gradient updates.
        IV  epochs (int): Number of epochs for training.
        V   batch_size (int): Size of the batches used in training.
        VI  lambda_val (float): Regularization parameter.
        VII momentum (float): Momentum term (for momentum-based optimizers).
        VIII seed (int): Random seed for reproducibility.

    Returns:
    ------------
        I   t_pred (np.ndarray): Predicted labels for the training set.
        II  t_train (np.ndarray): True labels for the training set.
        III accuracy (float): Training accuracy.
        IV  weights (np.ndarray): Optimized weight vector.
    """
    # Train logistic regression model
    weights = logistic_regression_sgd(X_train, t_train, scheduler=scheduler, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, lambda_val=lambda_val, mom=momentum)
    # Test model
    t_pred = predict(X_test, weights)
    t_test_labels = y_test
    accuracy = np.mean(t_pred == t_test_labels)

    return t_pred, t_test_labels, accuracy, weights


def predict(X, weights):
    """
    Description:
    ------------
        Generates predictions using the trained logistic regression model.

    Parameters:
    ------------
        I   X (np.ndarray): Design matrix with n rows (samples) and p columns (features).
        II  weights (np.ndarray): Optimized weight vector from training.
        III threshold (float): Decision threshold for binary classification (default: 0.5).

    Returns:
    ------------
        I   predictions (np.ndarray): Binary prediction vector (1 for hazardous class, 0 for non-hazardous class).
    """
    logits = X @ weights
    probabilities = sigmoid(logits)
    return (probabilities >= 0.5).astype(int)



def heatplot_eta_lmbda_log_reg(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler=None, epochs=100, batch_size=8, mom=0.0, title=None, savename=None):
    """
    Description:
    ------------
        Generates heatmaps showing the accuracy and recall for various learning
        rates (eta) and regularization strengths (lambda) using logistic regression.

    Parameters:
    ------------
        I   X_train, t_train (np.ndarray): Training features and labels.
        II  X_test, t_test (np.ndarray): Testing features and labels.
        III eta_values (list[float]): List of learning rates to test.
        IV  lambda_values (list[float]): List of regularization strengths to test.
        V   scheduler (str/class): Optimizer used for training (e.g., Adam, Adagrad, RMSprop).
        VI  epochs (int): Number of epochs for training.
        VII batch_size (int): Size of the batches used in training.
        VIII mom (float): Momentum term (for momentum-based optimizers).
        IX  title (str): Title variation for the heatmap and result summaries.
        X   savename (str): File name for saving the accuracy heatmap.

    Returns:
    ------------
        I   results_df (pd.DataFrame): Summary DataFrame containing mean and max recall for each class.
    """

    accuracy_matrix = np.zeros((len(eta_values), len(lambda_values)))
    recall_list_non_hazardous = []
    recall_list_hazardous = []

    for i, eta in enumerate(eta_values):
        print(f"\r\033[K Working on eta value {i+1}/{len(eta_values)} for scheduler: {title}.", end='')
        for j, lam in enumerate(lambda_values):
            t_pred, t_test_labels, accuracy, weights = train(X_train, t_train, scheduler=scheduler, learning_rate=eta, epochs=epochs, batch_size=batch_size, lambda_val=lam, momentum=mom)
            accuracy_matrix[i, j] = accuracy

            recall_values = recall_score(y_test, t_pred, average=None)
            recall_list_non_hazardous.append(recall_values[0])
            recall_list_hazardous.append(recall_values[1])
            #print(f"\nPrecision non hazard: {precision_list_non_hazardous} using eta={eta} and lambda={lam}.")
            #print(f"\nPrecision hazard: {precision_list_hazardous[-1]} using eta={eta} and lambda={lam}.")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_matrix, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=np.log10(lambda_values), yticklabels=eta_values, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 12}, vmin=0.5, vmax=1.0)
    plt.xlabel("log10(Lambda)", fontsize=14)
    plt.ylabel("Learning Rate (Eta)", fontsize=14)
    plt.title(f"Accuracy of Logistic Regression using {title}", fontsize=16)

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    #plt.show()

    # Plotting recall score for hazardous class
    recall_matrix = np.array(recall_list_hazardous).reshape(len(eta_values), len(lambda_values))

    plt.figure(figsize=(8, 6))
    sns.heatmap(recall_matrix, annot=True, fmt=".4f", xticklabels=np.log10(lambda_values), yticklabels=eta_values, cmap="viridis", annot_kws={"size": 12})
    plt.xlabel("log10(Lambda)", fontsize=14)
    plt.ylabel("Learning Rate (Eta)", fontsize=14)
    plt.title(f"Recall for Hazardous Class using {title}", fontsize=16)
    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"Recall_Hazard_{title}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")

    # Compute mean precision for each class
    mean_recall_class_0 = np.mean(recall_list_non_hazardous)
    mean_recall_class_1 = np.mean(recall_list_hazardous)

    # Compute max precision for each class
    max_recall_class_0 = np.max(recall_list_non_hazardous)
    max_recall_class_1 = np.max(recall_list_hazardous)

    # Create a DataFrame for the output
    results_df = pd.DataFrame({
        "Scheduler": [title],
        "Mean Class 0": [mean_recall_class_0],
        "Max Class 0": [max_recall_class_0],
        "Mean Class 1": [mean_recall_class_1],
        "Max Class 1": [max_recall_class_1]
    })

    # Set display options for pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    # Print results
    print("Recall Results Summary:")
    print(results_df.to_string(index=False))
    print("")

    return results_df




np.random.seed(123)
seed(123)

#Preprocess dataset
file_path = 'C:\\Users\\soren\\maskin\\proj3\\dataset\\dataset'
X_train, X_test, y_train, y_test = preprocess_dataset(file_path)



eta_values = [0.001, 0.01, 0.05, 0.1, 0.5]
#lambda_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_values = np.logspace(-5, -1, 5)

heatplot_eta_lmbda_log_reg(X_train, y_train, X_test, y_test, eta_values, lambda_values, scheduler=None, epochs=200, batch_size=16, title="no_scheduler", savename="log_reg_class_heatmap_eta_lambda_none.pdf")
heatplot_eta_lmbda_log_reg(X_train, y_train, X_test, y_test, eta_values, lambda_values, scheduler=None, epochs=200, batch_size=16, mom=0.9, title="momentum", savename="log_reg_class_heatmap_eta_lambda_mom.pdf")
heatplot_eta_lmbda_log_reg(X_train, y_train, X_test, y_test, eta_values, lambda_values, scheduler="adam", epochs=200, batch_size=16, title="Adam", savename="log_reg_class_heatmap_eta_lambda_adam.pdf")
heatplot_eta_lmbda_log_reg(X_train, y_train, X_test, y_test, eta_values, lambda_values, scheduler="ada", epochs=200, batch_size=16, title="Adagrad", savename="log_reg_class_heatmap_eta_lambda_ada.pdf")
heatplot_eta_lmbda_log_reg(X_train, y_train, X_test, y_test, eta_values, lambda_values, scheduler="rms", epochs=200, batch_size=16, title="RMSprop", savename="log_reg_class_heatmap_eta_lambda_rms.pdf")
