import os
from NN import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns


def heatplot_accuracy_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, scheduler_class, scheduler_params, hidden_func=sigmoid,
                                 output_func=identity, cost_func=CostOLS, learning_rate=0.0005,
                                 lambda_val=1e-5, epochs=100, seed=123, title=None, savename=None):

    print(f"Working on nodes/layer plot using {scheduler_class} and hidden function {hidden_func}")
    # Initialize result matrix for accuracy
    accuracy_results = np.zeros((len(neuron_options), len(layer_options)))

    np.random.seed(seed)

    for i, neurons in enumerate(neuron_options):
        for j, layers in enumerate(layer_options):
            # Set up network with current neurons and layers
            dimensions = (X_train.shape[1],) + (neurons,) * layers + (3,)
            network = FFNN(dimensions, hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            scheduler = scheduler_class(eta=learning_rate, **scheduler_params)

            # Train network
            scores = network.fit(X_train, t_train, scheduler, lam=lambda_val, epochs=epochs, X_val=X_test, t_val=t_test)

            # Store final validation accuracy
            accuracy_results[i, j] = scores['val_accs'][-1]  # Last epoch's accuracy

    # Plot heatmap for accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, fmt=".2f", cmap="viridis", xticklabels=layer_options,
                yticklabels=neuron_options, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 14}, vmin=0.8, vmax=1.0)
    plt.title(f"Accuracy for Nodes vs Layers using {title}", fontsize=16)
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Nodes per Layer", fontsize=14)


    save_dir = "figs"
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

    return accuracy_results




def heatplot_accuracy_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, scheduler_class,
                                 scheduler_params, neurons=20, layers=1, hidden_func=sigmoid, output_func=softmax,
                                 cost_func=CostCrossEntropy, epochs=100, seed=123, title=None, savename=None):

    # Initialize result matrix for accuracy
    accuracy_results = np.zeros((len(eta_values), len(lambda_values)))

    np.random.seed(seed)

    for i, eta in enumerate(eta_values):
        for j, lam in enumerate(lambda_values):

            dimensions = (X_train.shape[1],) + (neurons,) * layers + (3,)
            network = FFNN(dimensions, hidden_func=hidden_func, output_func=output_func, cost_func=cost_func, seed=seed)
            network.reset_weights()

            scheduler = scheduler_class(eta=eta, **scheduler_params)

            # Train network
            scores = network.fit(X_train, t_train, scheduler, lam=lam, epochs=epochs, X_val=X_test, t_val=t_test)

            # Store final accuracy
            accuracy_results[i, j] = scores['val_accs'][-1]  # Last epoch's accuracy

    # Plot heatmap for accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, fmt=".2f", cmap="viridis", xticklabels=np.log10(lambda_values),
                yticklabels=eta_values, cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 14}, vmin=0.8, vmax=1.0)
    plt.title(f"Accuracy for Eta vs Lambda using {title}", fontsize=16)
    plt.xlabel("log10(Lambda)", fontsize=14)
    plt.ylabel("Eta", fontsize=14)

    save_dir = "figs"
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

    return accuracy_results




np.random.seed(123)
seed(123)

wine = load_wine()

inputs = wine.data
outputs = wine.target
outputs = outputs.reshape(outputs.shape[0], 1)

labels = wine.feature_names


X_train, X_test, t_train, t_test = train_test_split(inputs, outputs, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#one hot encode outputs
onehot = OneHotEncoder(sparse=False)
t_train = onehot.fit_transform(t_train)
t_test = onehot.transform(t_test)





# Amount of neurons per layer and layers
neuron_options = [10, 20, 40, 80, 140, 200]
layer_options = [1, 2, 3, 4, 5, 6]

scheduler_params_ada = {}
scheduler_params_rmsprop = {"rho": 0.9}
scheduler_params_adam = {"rho": 0.9, "rho2":0.999}


#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what amount of nodes per layer, and amount of layers provide best accuracy
heatplot_accuracy_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=sigmoid,
                                 output_func=softmax, cost_func=CostCrossEntropy, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="sigmoid", savename="Heatplot_accuracy_nodes_layers_class_sigmoid.pdf")


heatplot_accuracy_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=RELU,
                                 output_func=softmax, cost_func=CostCrossEntropy, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="RELU", savename="Heatplot_accuracy_nodes_layers_class_RELU.pdf")


heatplot_accuracy_nodes_layers(X_train, t_train, X_test, t_test, neuron_options,
                                 layer_options, Adam, scheduler_params_adam, hidden_func=LRELU,
                                 output_func=softmax, cost_func=CostCrossEntropy, learning_rate=0.01,
                                 lambda_val=1e-5, epochs=100, seed=123, title="LRELU", savename="Heatplot_accuracy_nodes_layers_class_LRELU.pdf")



eta_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]
lambda_values = np.logspace(-5, -1, 5)

#Making heatplots using sigmoid, RELU and LRELU activation functions for the hidden layers
#Studying what eta and lambda values provide best accuracy

heatplot_accuracy_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                                 scheduler_params_adam, neurons=20, layers=1, hidden_func=sigmoid, output_func=softmax,
                                 cost_func=CostCrossEntropy, epochs=100, seed=123, title="sigmoid", savename="Heatplot_accuracy_eta_lmbda_class_sigmoid.pdf")


heatplot_accuracy_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                                 scheduler_params_adam, neurons=20, layers=1, hidden_func=RELU, output_func=softmax,
                                 cost_func=CostCrossEntropy, epochs=100, seed=123, title="RELU", savename="Heatplot_accuracy_eta_lmbda_class_RELU.pdf")


heatplot_accuracy_eta_lambda(X_train, t_train, X_test, t_test, eta_values, lambda_values, Adam,
                                 scheduler_params_adam, neurons=20, layers=1, hidden_func=LRELU, output_func=softmax,
                                 cost_func=CostCrossEntropy, epochs=100, seed=123, title="LRELU", savename="Heatplot_accuracy_eta_lmbda_class_LRELU.pdf")
