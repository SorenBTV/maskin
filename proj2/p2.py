import os
from NN import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from functions import *
from activation_functions import *




def heatplot_MSE_R2_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, scheduler_class, scheduler_params, hidden_func=sigmoid,
                                 output_func=identity, cost_func=CostOLS, learning_rate=0.01,
                                 lambda_val=0.0001, epochs=100, seed=123, title=None, savename=None):

    # Initialize result matrices
    mse_results = np.zeros((len(neuron_options), len(layer_options)))  # For number of neurons vs layers
    r2_results = np.zeros((len(neuron_options), len(layer_options)))

    # Set random seed
    np.random.seed(seed)

    for i, neurons in enumerate(neuron_options):
        for j, layers in enumerate(layer_options):
            print(f"\n Training with {neurons} neurons and {layers} layers")

            # Reset and initialize network with current number of neurons and layers
            network = FFNN((X_train.shape[1],) + (neurons,) * layers + (1,), hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            # Initialize the scheduler dynamically based on the passed class and params
            scheduler = scheduler_class(eta=learning_rate, **scheduler_params)

            # Train the network
            train_scores = network.fit(X_train, t_train, scheduler, lam=lambda_val, epochs=epochs)
            test_predictions = network.predict(X_test)

            # Compute MSE and R² for the test set
            mse = mean_squared_error(t_test, test_predictions)
            r2 = r2_score(t_test, test_predictions)

            # Store results
            mse_results[i, j] = mse
            r2_results[i, j] = r2


    # Create side-by-side subplots for MSE and R² heatmaps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))  # Adjust figsize as needed

    # Plot MSE heatmap for eta vs lambda
    sns.heatmap(mse_results, annot=True, fmt=".3f", cmap="magma", xticklabels=layer_options, yticklabels=neuron_options,
                cbar_kws={'label': 'MSE'}, annot_kws={"size": 14}, ax=ax1)
    ax1.set_title(f"MSE for neurons per layer vs. layers using {title}", fontsize=16)
    ax1.set_xlabel("Neurons per layer", fontsize=14)
    ax1.set_ylabel("Layers", fontsize=14)

    # Plot R² heatmap for eta vs lambda
    sns.heatmap(r2_results, annot=True, fmt=".2f", cmap="magma", xticklabels=layer_options, yticklabels=neuron_options,
                cbar_kws={'label': 'R²'}, annot_kws={"size": 14}, ax=ax2, vmin=0, vmax=1)
    ax2.set_title(f"R^2-score for neurons per layer vs. layers using {title}", fontsize=16)
    ax2.set_xlabel("Neurons per layer", fontsize=14)
    ax2.set_ylabel("Layers", fontsize=14)

    plt.tight_layout()  # Adjust layout to avoid overlap

    plt.subplots_adjust(hspace=0.15)
    save_dir = "figs"
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    #plt.show()

    return mse_results, r2_results



def heatplot_MSE_R2_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler_class,
                               scheduler_params, neurons=80, layers=4, hidden_func=None, output_func=None,
                               cost_func=None, epochs=100, seed=123, title=None, savename=None):
    # Initialize result matrices
    mse_results = np.zeros((len(eta_values), len(lambda_values)))
    r2_results = np.zeros((len(eta_values), len(lambda_values)))

    for i, eta in enumerate(eta_values):
        for j, lam in enumerate(lambda_values):
            #print(f"Training with eta={eta} and lambda={lam}")

            # Reset and initialize network with current eta and lambda
            network = FFNN((X_train.shape[1],) + (neurons,) * layers + (1,), hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            # Set scheduler with the current eta
            scheduler = scheduler_class(eta=eta, **scheduler_params)

            # Train the network
            train_scores = network.fit(X_train, t_train, scheduler, lam=lam, epochs=epochs)
            test_predictions = network.predict(X_test)

            # Compute MSE and R² for the test set
            mse = mean_squared_error(t_test, test_predictions)
            r2 = r2_score(t_test, test_predictions)

            # Store results
            mse_results[i, j] = mse
            r2_results[i, j] = r2

    # Create side-by-side subplots for MSE and R² heatmaps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))  # Adjust figsize as needed

    # Plot MSE heatmap for eta vs lambda
    sns.heatmap(mse_results, annot=True, fmt=".3f", cmap="magma", xticklabels=np.log10(lambda_values), yticklabels=eta_values,
                cbar_kws={'label': 'MSE'}, annot_kws={"size": 14}, ax=ax1)
    ax1.set_title(f"MSE for eta vs lambda using {title}", fontsize=16)
    ax1.set_xlabel("Lambda", fontsize=14)
    ax1.set_ylabel("Eta", fontsize=14)

    # Plot R² heatmap for eta vs lambda
    sns.heatmap(r2_results, annot=True, fmt=".2f", cmap="magma", xticklabels=np.log10(lambda_values), yticklabels=eta_values,
                cbar_kws={'label': 'R²'}, annot_kws={"size": 14}, ax=ax2, vmin=0, vmax=1)
    ax2.set_title(f"R^2-score for eta vs lambda using {title}", fontsize=16)
    ax2.set_xlabel("Lambda", fontsize=14)
    ax2.set_ylabel("Eta", fontsize=14)

    plt.tight_layout()

    plt.subplots_adjust(hspace=0.15)
    save_dir = "figs"
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    #plt.show()

    return mse_results, r2_results




np.random.seed(123)
seed(123)

#Setting up data using the Franke Function
n = 100
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
x, y = x.ravel(), y.ravel()

target = np.ravel(FrankeFunction(x, y))
target = target.reshape(target.shape[0], 1)

poly_degree = 3
X = create_design_matrix(x, y, poly_degree)

X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.3, random_state=42)



# Amount of neurons per layer and layers
neuron_options = [5, 10, 20, 40, 80]
layer_options = [1, 2, 3, 4, 5, 6]

scheduler_params_ada = {}
scheduler_params_rmsprop = {"rho": 0.9}
scheduler_params_adam = {"rho": 0.9, "rho2":0.999}

#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what amount of nodes per layer, and amount of layers provide best MSE and R2
heatplot_MSE_R2_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=sigmoid,
                                 output_func=identity, cost_func=CostOLS, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="Sigmoid", savename="Heatplot_MSE_R2_neurons_layers_sigmoid.pdf")

heatplot_MSE_R2_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=RELU,
                                 output_func=identity, cost_func=CostOLS, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="RELU", savename="Heatplot_MSE_R2_neurons_layers_RELU.pdf")

heatplot_MSE_R2_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=LRELU,
                                 output_func=identity, cost_func=CostOLS, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="LRELU", savename="Heatplot_MSE_R2_neurons_layers_LRELU.pdf")




#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what eta and lambda values provide best MSE and R2
eta_values = [0.001, 0.005, 0.01, 0.05, 0.1]
lambda_values = np.logspace(-5, -1, 5)
heatplot_MSE_R2_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                               scheduler_params_adam, neurons=20, layers=1, hidden_func=sigmoid, output_func=identity,
                               cost_func=CostOLS, epochs=100, seed=123, title="Sigmoid", savename="Heatplot_MSE_R2_eta_lambda_sigmoid.pdf")


heatplot_MSE_R2_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                               scheduler_params_adam, neurons=20, layers=1, hidden_func=RELU, output_func=identity,
                               cost_func=CostOLS, epochs=100, seed=123, title="RELU", savename="Heatplot_MSE_R2_eta_lambda_RELU.pdf")


heatplot_MSE_R2_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                               scheduler_params_adam, neurons=20, layers=1, hidden_func=LRELU, output_func=identity,
                               cost_func=CostOLS, epochs=100, seed=123, title="LRELU", savename="Heatplot_MSE_R2_eta_lambda_LRELU.pdf")
