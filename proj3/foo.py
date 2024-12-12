import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score
from NN import *
from scipy.io import arff
import seaborn as sns


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

    return X_train, X_test, y_train, y_test



def heatplot_accuracy_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, scheduler_class, scheduler_params, hidden_func=sigmoid,
                                 output_func=sigmoid, cost_func=CostCrossEntropy, learning_rate=0.001,
                                 lambda_val=1e-5, epochs=100, seed=123, title=None, savename=None):
    """
    Description:
    ------------
        Generates a heatmap showing the accuracy and recall for varying
        numbers of neurons per layer and layers in a neural network.

    Parameters:
    ------------
        I   X_train, t_train (np.ndarray): Training features and labels.
        II  X_test, t_test (np.ndarray): Testing features and labels.
        III neuron_options (list[int]): List of neuron counts to test.
        IV  layer_options (list[int]): List of layer counts to test.
        V   scheduler_class (class): Scheduler used for optimization.
        VI  scheduler_params (dict): Parameters for the scheduler.
        VII hidden_func, output_func (Callable): Activation functions for hidden and output layers.
        VIII cost_func (Callable): Cost function for the model.
        IX  learning_rate (float): Learning rate for the optimizer.
        X   lambda_val (float): Regularization parameter.
        XI  epochs (int): Number of training epochs.
        XII seed (int): Random seed for reproducibility.
        XIII title (str): Title variation for the heatmaps.
        XIV savename (str): Name of the file to save the heatmaps.

    Returns:
    ------------
        I   accuracy_results (np.ndarray): Matrix containing accuracy results.
    """

    print(f"Working on nodes/layer plot using {scheduler_class} and hidden function {hidden_func}")
    # Initialize result matrices for accuracy and recall score
    accuracy_results = np.zeros((len(neuron_options), len(layer_options)))
    recall_results = np.zeros((len(neuron_options), len(layer_options)))
    np.random.seed(seed)

    for i, neurons in enumerate(neuron_options):
        for j, layers in enumerate(layer_options):
            dimensions = (X_train.shape[1],) + (neurons,) * layers + (1,)
            network = FFNN(dimensions, hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            scheduler = scheduler_class(eta=learning_rate, **scheduler_params)

            # Train network
            scores = network.fit(X_train, t_train, scheduler, lam=lambda_val, epochs=epochs, X_val=X_test, t_val=t_test)

            # Store final accuracy
            accuracy_results[i, j] = scores['val_accs'][-1]  # Last epoch's accuracy

            # Predictions and recall score
            y_pred = network.predict(X_test)
            y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
            recall_values = recall_score(t_test.flatten(), y_pred_binary, average=None, zero_division=0)

            recall_results[i, j] = recall_values[1]



    # Plot heatmap for accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, fmt=".2f", cmap="viridis", xticklabels=layer_options,
                yticklabels=neuron_options, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 14}, vmin=0.2, vmax=1.0)
    plt.title(f"Accuracy for Nodes vs Layers using {title}", fontsize=16)
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Nodes per Layer", fontsize=14)


    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


    # Plot heatmap for recall score of class 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(recall_results, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=layer_options, yticklabels=neuron_options, cbar_kws={'label': 'Class 1 recall'},
                annot_kws={"size": 12})
    plt.title(f"Class 1 Recall score for Nodes vs Layers using {title}", fontsize=16)
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Nodes per Layer", fontsize=14)

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"NN_neuron_layer_recall_hazard_{title}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


    return accuracy_results




def heatplot_accuracy_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler_class,
                                 scheduler_params, neurons=20, layers=1, hidden_func=sigmoid, output_func=softmax,
                                 cost_func=CostCrossEntropy, epochs=100, seed=123, title=None, savename=None):

    """
    Description:
    ------------
        Generates a heatmap of accuracy and recall for varying learning rates
        (eta) and regularization strengths (lambda) using a neural network.

    Parameters:
    ------------
        I   X_train, t_train (np.ndarray): Training features and labels.
        II  X_test, t_test (np.ndarray): Testing features and labels.
        III eta_options (list[float]): List of learning rates to test.
        IV  lambda_options (list[float]): List of regularization strengths to test.
        V   scheduler_class (class): Scheduler class used for optimization.
        VI  scheduler_params (dict): Parameters for the scheduler.
        VII hidden_func, output_func (Callable): Activation functions for the hidden and output layers.
        VIII cost_func (Callable): Cost function to optimize the model.
        IX  neurons (int): Number of neurons per layer.
        X   layers (int): Number of hidden layers.
        XI  epochs (int): Number of epochs to train for.
        XII seed (int): Random seed for reproducibility.
        XIII title (str): Title variation for the heatmaps.
        XIV savename (str): File name to save the heatmaps.

    Returns:
    ------------
        I   accuracy_results (np.ndarray): Matrix containing accuracy results.
    """

    # Initialize result matrix for accuracy and recall score
    accuracy_results = np.zeros((len(eta_values), len(lambda_values)))
    recall_results = np.zeros((len(eta_values), len(lambda_values)))
    np.random.seed(seed)

    for i, eta in enumerate(eta_values):
        for j, lam in enumerate(lambda_values):

            dimensions = (X_train.shape[1],) + (neurons,) * layers + (1,)
            network = FFNN(dimensions, hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            scheduler = scheduler_class(eta=eta, **scheduler_params)

            # Train network
            scores = network.fit(X_train, t_train, scheduler, lam=lam, epochs=epochs, X_val=X_test, t_val=t_test)

            # Store final accuracy
            accuracy_results[i, j] = scores['val_accs'][-1]  # Last epoch's accuracy

            # Predictions and recall score
            y_pred = network.predict(X_test)
            y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
            recall_values = recall_score(t_test.flatten(), y_pred_binary, average=None, zero_division=0)

            #Store recall score values for further analysis
            recall_results[i, j] = recall_values[1]


    # Plot heatmap for accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, fmt=".2f", cmap="viridis", xticklabels=np.log10(lambda_values),
                yticklabels=eta_values, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 14}, vmin=0.2, vmax=1.0)
    plt.title(f"Accuracy for Eta vs Lambda using {title}", fontsize=16)
    plt.xlabel("log10(Lambda)", fontsize=14)
    plt.ylabel("Learning rate (Eta)", fontsize=14)

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")


    # Plot heatmap for recall score of class 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(recall_results, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=np.log10(lambda_values), yticklabels=eta_values, cbar_kws={'label': 'Class 1 recall'},
                annot_kws={"size": 12})
    plt.title(f"Class 1 recall for Eta vs Lambda using {title}", fontsize=16)
    plt.xlabel("log10(Lambda)", fontsize=14)
    plt.ylabel("Learning rate (Eta)", fontsize=14)

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"NN_eta_lam_recall_hazard_{title}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

    return accuracy_results




#Preprocess dataset
file_path = 'C:\\Users\\soren\\maskin\\proj3\\dataset\\dataset'
X_train, X_test, y_train, y_test = preprocess_dataset(file_path)
# Reshape y to be 2D, as required in FFNN
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


# Amount of neurons per layer and layers
neuron_options = [4, 10, 20, 40, 60]
layer_options = [1, 2, 3, 4, 5, 6, 7]

#Ready scheduler parameters
scheduler_params_none = {}
scheduler_params_mom = {"momentum":0.9}
scheduler_params_ada = {}
scheduler_params_rmsprop = {"rho": 0.9}
scheduler_params_adam = {"rho": 0.9, "rho2":0.999}


#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what amount of nodes per layer, and amount of layers provide best accuracy and recall score
heatplot_accuracy_nodes_layers(X_train, y_train, X_test, y_test, neuron_options,
                                 layer_options, Constant, scheduler_params_none, hidden_func=sigmoid,
                                 output_func=sigmoid, cost_func=CostCrossEntropy, learning_rate=1e-3,
                                 lambda_val=1e-4, epochs=200, seed=123, title="sigmoid", savename="Heatplot_accuracy_nodes_layers_class_sigmoid.pdf")


heatplot_accuracy_nodes_layers(X_train, y_train, X_test, y_test, neuron_options,
                                 layer_options, Constant, scheduler_params_none, hidden_func=RELU,
                                 output_func=sigmoid, cost_func=CostCrossEntropy, learning_rate=1e-3,
                                 lambda_val=1e-4, epochs=200, seed=123, title="RELU", savename="Heatplot_accuracy_nodes_layers_class_RELU.pdf")


heatplot_accuracy_nodes_layers(X_train, y_train, X_test, y_test, neuron_options,
                                 layer_options, Constant, scheduler_params_none, hidden_func=LRELU,
                                 output_func=sigmoid, cost_func=CostCrossEntropy, learning_rate=1e-3,
                                 lambda_val=1e-4, epochs=200, seed=123, title="LRELU", savename="Heatplot_accuracy_nodes_layers_class_LRELU.pdf")




eta_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5]
lambda_values = np.logspace(-5, -1, 5)
print(lambda_values)

#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what eta and lambda values provide best accuracy and recall score

heatplot_accuracy_eta_lambda(X_train, y_train, X_test, y_test, eta_values, lambda_values, Constant,
                                 scheduler_params_none, neurons=20, layers=1, hidden_func=sigmoid, output_func=sigmoid,
                                 cost_func=CostCrossEntropy, epochs=200, seed=123, title="sigmoid", savename="Heatplot_accuracy_eta_lmbda_class_sigmoid.pdf")


heatplot_accuracy_eta_lambda(X_train, y_train, X_test, y_test, eta_values, lambda_values, Constant,
                                 scheduler_params_none, neurons=4, layers=2, hidden_func=RELU, output_func=sigmoid,
                                 cost_func=CostCrossEntropy, epochs=200, seed=123, title="RELU", savename="Heatplot_accuracy_eta_lmbda_class_RELU.pdf")


heatplot_accuracy_eta_lambda(X_train, y_train, X_test, y_test, eta_values, lambda_values, Constant,
                                 scheduler_params_none, neurons=4, layers=2, hidden_func=LRELU, output_func=sigmoid,
                                 cost_func=CostCrossEntropy, epochs=200, seed=123, title="LRELU", savename="Heatplot_accuracy_eta_lmbda_class_LRELU.pdf")
