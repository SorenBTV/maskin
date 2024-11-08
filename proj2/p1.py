import os
import numpy as np
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from cost import *
from functions import *
from scheduler import *


def CostOLS(theta,y,X):
    n = len(y)
    return (1.0/n)*anp.sum((y-X @ theta)**2)


def CostRidge(theta, y, X, lmb):
    n = len(y)
    return (1.0/n)*anp.sum((y-X @ theta)**2)+lmb*theta.T@theta

def GD(X, y, M=1, epochs=100, scheduler=None, learning_rate=0.01, momentum=0.0, lmb=1e-5, Ridge=False):
    n = len(y)
    m = int(n/M)
    theta = np.random.randn(3,1)

    #scheduler initialization
    if scheduler=="ada":
        scheduler = Adagrad(eta=learning_rate)
    elif scheduler=="rms":
        scheduler = RMS_prop(eta=learning_rate, rho=0.9)
    elif scheduler=="adam":
        scheduler = Adam(eta=learning_rate, rho=0.9, rho2=0.999)

    mse_per_epoch = []
    r2_per_epoch = []

    if Ridge:
        theta_linreg = Ridge_fit_beta(X, y, lmb)
        training_gradient = grad(lambda t, y, X: CostRidge(t, y, X, lmb))
    else:
        theta_linreg = OLS_fit_beta(X, y)
        training_gradient = grad(CostOLS)

    for epoch in range(epochs):
        np.random.seed(123 + epoch)
        change = 0
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0 / M) * training_gradient(theta, yi, xi) + momentum * change

            #Use scheduler to update gradient if scheduler is chosen.
            if scheduler is not None:
                change = scheduler.update_change(gradients)
            else:
                change = learning_rate * gradients

            theta -= change

        Xnew_this_epoch = np.c_[np.ones((n,1)), np.linspace(0, 1, n), np.linspace(0, 1, n)**2]
        ypredict_this_epoch = Xnew_this_epoch @ theta
        mse_per_epoch.append(mean_squared_error(y, ypredict_this_epoch))
        r2_per_epoch.append(r2_score(y, ypredict_this_epoch))




    return theta, mse_per_epoch, r2_per_epoch



def SGD(X, y, M=8, epochs=100, scheduler=None, learning_rate=0.01, momentum=0.0, lmb=1e-5, Ridge=False):
    n = len(y)
    m = int(n/M)
    theta = np.random.randn(3,1)

    #scheduler initialization
    if scheduler=="ada":
        scheduler = Adagrad(eta=learning_rate)
    elif scheduler=="rms":
        scheduler = RMS_prop(eta=learning_rate, rho=0.9)
    elif scheduler=="adam":
        scheduler = Adam(eta=learning_rate, rho=0.9, rho2=0.999)

    mse_per_epoch = []
    r2_per_epoch = []

    if Ridge:
        theta_linreg = Ridge_fit_beta(X, y, lmb)
        training_gradient = grad(lambda t, y, X: CostRidge(t, y, X, lmb))

    else:
        theta_linreg = OLS_fit_beta(X, y)
        training_gradient = grad(CostOLS)


    for epoch in range(epochs):
        np.random.seed(123 + epoch)
        change = 0
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (1.0 / M) * training_gradient(theta, yi, xi) + momentum * change

            #Use scheduler to update gradient if scheduler is chosen.
            if scheduler is not None:
                change = scheduler.update_change(gradients)
            else:
                change = learning_rate * gradients

            theta -= change

        Xnew_this_epoch = np.c_[np.ones((n,1)), np.linspace(0, 1, n), np.linspace(0, 1, n)**2]
        ypredict_this_epoch = Xnew_this_epoch @ theta
        mse_per_epoch.append(mean_squared_error(y, ypredict_this_epoch))
        r2_per_epoch.append(r2_score(y, ypredict_this_epoch))



    return theta, mse_per_epoch, r2_per_epoch

def plot_mse_r2(mse_ada, mse_rms, mse_adam, mse_none, mse_none_mom, r2_ada, r2_rms, r2_adam, r2_none, r2_none_mom, title=None, savename=None, epochs=100):

    plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), mse_ada, label="Adagrad")
    plt.plot(range(epochs), mse_rms, label="RMSprop")
    plt.plot(range(epochs), mse_adam, label="Adam")
    plt.plot(range(epochs), mse_none, label="No scheduler without momentum")
    plt.plot(range(epochs), mse_none_mom, label="No scheduler with momentum")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"MSE Over Epochs using {title}")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), r2_ada, label="Adagrad")
    plt.plot(range(epochs), r2_rms, label="RMSprop")
    plt.plot(range(epochs), r2_adam, label="Adam")
    plt.plot(range(epochs), r2_none, label="No scheduler without momentum")
    plt.plot(range(epochs), r2_none_mom, label="No scheduler with momentum")
    plt.xlabel("Epochs")
    plt.ylabel("R²")
    plt.title(f"R² Over Epochs using {title}")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, savename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    #plt.show()

    return 0


def plot_batch_size(batch_list, epochs=100, momentum=0.0, scheduler=None, learning_rate=0.01, Ridge=False, title=None):
    print(f"Making plots comparing batch sizes using scheduler:{title}")

    plt.figure()
    for i, batch_size in enumerate(batch_list):
        theta, mse, r2 = SGD(X, y, M=batch_size, momentum=momentum, scheduler=scheduler, learning_rate=learning_rate, Ridge=Ridge)
        plt.plot(range(epochs), mse, label=f"batch_size={batch_size}")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"{title}")
    plt.legend()
    plt.grid()

    save_dir = "figs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"Batch_size_study_{title}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    #plt.show()

    return 0

#Initializing function
np.random.seed(123)
n = 100
x = np.linspace(0, 1, n)
X = np.c_[np.ones((100,1)), x, x*x]
y = 4*X[:,0] + 3*X[:,1] + 2*X[:,2]
#y = 2*X[:,2]


etas = [0.001, 0.005, 0.01, 0.1, 0.3]
lambdas = np.logspace(-5, -1, 5)


mse_matrix_gd_ada = np.zeros((len(etas), len(lambdas)))
mse_matrix_sgd_ada = np.zeros((len(etas), len(lambdas)))

mse_matrix_gd_rms = np.zeros((len(etas), len(lambdas)))
mse_matrix_sgd_rms = np.zeros((len(etas), len(lambdas)))

mse_matrix_gd_adam = np.zeros((len(etas), len(lambdas)))
mse_matrix_sgd_adam = np.zeros((len(etas), len(lambdas)))

mse_matrix_gd_none = np.zeros((len(etas), len(lambdas)))
mse_matrix_sgd_none = np.zeros((len(etas), len(lambdas)))

mse_matrix_gd_mom = np.zeros((len(etas), len(lambdas)))
mse_matrix_sgd_mom = np.zeros((len(etas), len(lambdas)))

print(f"Working on eta value heatplot comparison of schedulers...")
for i, eta in enumerate(etas):
    for j, lmb in enumerate(lambdas):
        #Adagrad
        theta_gd_ada, mse_per_epoch_gd_ada, r2_per_epoch_gd_ada = GD(X, y, scheduler="ada", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_gd_ada[i, j] = mse_per_epoch_gd_ada[-1]

        theta_sgd_ada, mse_per_epoch_sgd_ada, r2_per_epoch_sgd_ada = SGD(X, y, scheduler="ada", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_sgd_ada[i, j] = mse_per_epoch_sgd_ada[-1]

        #RMS
        theta_gd_rms, mse_per_epoch_gd_rms, r2_per_epoch_gd_rms = GD(X, y, scheduler="rms", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_gd_rms[i, j] = mse_per_epoch_gd_rms[-1]

        theta_sgd_rms, mse_per_epoch_sgd_rms, r2_per_epoch_sgd_rms = SGD(X, y, scheduler="rms", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_sgd_rms[i, j] = mse_per_epoch_sgd_rms[-1]

        #Adam
        theta_gd_ada, mse_per_epoch_gd_ada, r2_per_epoch_gd_ada = GD(X, y, scheduler="adam", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_gd_ada[i, j] = mse_per_epoch_gd_ada[-1]

        theta_sgd_ada, mse_per_epoch_sgd_ada, r2_per_epoch_sgd_ada = SGD(X, y, scheduler="adam", learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_sgd_ada[i, j] = mse_per_epoch_sgd_ada[-1]

        #No scheduler
        theta_gd_none, mse_per_epoch_gd_none, r2_per_epoch_gd_none = GD(X, y, scheduler=None, learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_gd_none[i, j] = mse_per_epoch_gd_none[-1]

        theta_sgd_none, mse_per_epoch_sgd_none, r2_per_epoch_sgd_none = SGD(X, y, scheduler=None, learning_rate=eta, lmb=lmb, Ridge=True)
        mse_matrix_sgd_none[i, j] = mse_per_epoch_sgd_none[-1]

        #No scheduler with momentum
        theta_gd_mom, mse_per_epoch_gd_mom, r2_per_epoch_gd_mom = GD(X, y, scheduler=None, learning_rate=eta, momentum=0.9, lmb=lmb, Ridge=True)
        mse_matrix_gd_mom[i, j] = mse_per_epoch_gd_mom[-1]

        theta_sgd_mom, mse_per_epoch_sgd_mom, r2_per_epoch_sgd_mom = SGD(X, y, scheduler=None, learning_rate=eta, momentum=0.9, lmb=lmb, Ridge=True)
        mse_matrix_sgd_mom[i, j] = mse_per_epoch_sgd_mom[-1]

fig, ax = plt.subplots(5, 2, figsize=(16, 24))
ax = ax.flatten()
# Plot the Adagrad heatmaps
sns.heatmap(mse_matrix_gd_ada, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[0], vmin=0, vmax=0.5)
ax[0].set_title("GD Adagrad")
ax[0].set_xlabel("log10(λ)")
ax[0].set_ylabel("η (Learning Rate)")
sns.heatmap(mse_matrix_sgd_ada, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[1], vmin=0, vmax=25)
ax[1].set_title("SGD Adagrad")
ax[1].set_xlabel("log10(λ)")
ax[1].set_ylabel("η (Learning Rate)")

# Plot the RMS heatmaps
sns.heatmap(mse_matrix_gd_rms, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[2], vmin=0, vmax=0.35)
ax[2].set_title("GD RMS")
ax[2].set_xlabel("log10(λ)")
ax[2].set_ylabel("η (Learning Rate)")
sns.heatmap(mse_matrix_sgd_rms, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[3], vmin=0, vmax=0.35)
ax[3].set_title("SGD RMS")
ax[3].set_xlabel("log10(λ)")
ax[3].set_ylabel("η (Learning Rate)")

# Plot the Adam heatmaps
sns.heatmap(mse_matrix_gd_adam, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[4], vmin=0, vmax=0.1)
ax[4].set_title("GD Adam")
ax[4].set_xlabel("log10(λ)")
ax[4].set_ylabel("η (Learning Rate)")
sns.heatmap(mse_matrix_sgd_adam, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[5], vmin=0, vmax=0.1)
ax[5].set_title("SGD Adam")
ax[5].set_xlabel("log10(λ)")
ax[5].set_ylabel("η (Learning Rate)")

# Plot the none-scheduler heatmaps
sns.heatmap(mse_matrix_gd_none, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[6], vmin=0, vmax=0.175)
ax[6].set_title("GD no scheduler")
ax[6].set_xlabel("log10(λ)")
ax[6].set_ylabel("η (Learning Rate)")
sns.heatmap(mse_matrix_sgd_none, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[7], vmin=0, vmax=0.08)
ax[7].set_title("SGD no scheduler")
ax[7].set_xlabel("log10(λ)")
ax[7].set_ylabel("η (Learning Rate)")

# Plot the none-scheduler heatmaps with momentum
sns.heatmap(mse_matrix_gd_mom, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[8], vmin=0, vmax=0.4)
ax[8].set_title("GD no scheduler /w momentum")
ax[8].set_xlabel("log10(λ)")
ax[8].set_ylabel("η (Learning Rate)")
sns.heatmap(mse_matrix_sgd_mom, xticklabels=np.log10(lambdas), yticklabels=etas, cbar_kws={'label': 'MSE'}, cmap='magma', annot=True, fmt=".5f", ax=ax[9], vmin=0, vmax=0.05)
ax[9].set_title("SGD no scheduler /w momentum")
ax[9].set_xlabel("log10(λ)")
ax[9].set_ylabel("η (Learning Rate)")


plt.subplots_adjust(hspace=0.4)
save_dir = "figs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, "GD_SGD_heatmaps_5x2.pdf")
plt.savefig(save_path, bbox_inches='tight')
print(f"Figure saved to {save_path}")
#plt.show()



#print("Working on MSE and R2 plots...")

#Running and plotting MSE and R2 using SGD with OLS
theta_adagrad, mse_adagrad, r2_adagrad = SGD(X, y, scheduler="ada", Ridge=False)
theta_rmsprop, mse_rmsprop, r2_rmsprop = SGD(X, y, scheduler="rms", Ridge=False)
theta_adam, mse_adam, r2_adam = SGD(X, y, scheduler="adam", Ridge=False)
theta_none, mse_none, r2_none = SGD(X, y, Ridge=False)
theta_none_mom, mse_none_mom, r2_none_mom = SGD(X, y, momentum=0.9, Ridge=False)
plot_mse_r2(mse_adagrad, mse_rmsprop, mse_adam, mse_none, mse_none_mom, r2_adagrad, r2_rmsprop, r2_adam, r2_none, r2_none_mom, title="SGD with OLS", savename="Stochastic_gradient_descent_using_OLS.pdf")

#Running and plotting MSE and R2 using SGD with Ridge
theta_adagrad, mse_adagrad, r2_adagrad = SGD(X, y, scheduler="ada", Ridge=True)
theta_rmsprop, mse_rmsprop, r2_rmsprop = SGD(X, y, scheduler="rms", Ridge=True)
theta_adam, mse_adam, r2_adam = SGD(X, y, scheduler="adam", Ridge=True)
theta_none, mse_none, r2_none = SGD(X, y, Ridge=True)
theta_none_mom, mse_none_mom, r2_none_mom = SGD(X, y, momentum=0.9, Ridge=True)
plot_mse_r2(mse_adagrad, mse_rmsprop, mse_adam, mse_none, mse_none_mom, r2_adagrad, r2_rmsprop, r2_adam, r2_none, r2_none_mom, title="SGD with Ridge", savename="Stochastic_gradient_descent_using_Ridge.pdf")

#Running and plotting MSE and R2 using GD with OLS
theta_adagrad, mse_adagrad, r2_adagrad = GD(X, y, scheduler="ada", Ridge=False)
theta_rmsprop, mse_rmsprop, r2_rmsprop = GD(X, y, scheduler="rms", Ridge=False)
theta_adam, mse_adam, r2_adam = GD(X, y, scheduler="adam", Ridge=False)
theta_none, mse_none, r2_none = GD(X, y, Ridge=False)
theta_none_mom, mse_none_mom, r2_none_mom = GD(X, y, momentum=0.9, Ridge=False)
plot_mse_r2(mse_adagrad, mse_rmsprop, mse_adam, mse_none, mse_none_mom, r2_adagrad, r2_rmsprop, r2_adam, r2_none, r2_none_mom, title="GD with OLS", savename="Gradient_descent_using_OLS.pdf")

#Running and plotting MSE and R2 using GD with Ridge
theta_adagrad, mse_adagrad, r2_adagrad = GD(X, y, scheduler="ada", Ridge=True)
theta_rmsprop, mse_rmsprop, r2_rmsprop = GD(X, y, scheduler="rms", Ridge=True)
theta_adam, mse_adam, r2_adam = GD(X, y, scheduler="adam", Ridge=True)
theta_none, mse_none, r2_none = GD(X, y, Ridge=True)
theta_none_mom, mse_none_mom, r2_none_mom = GD(X, y, momentum=0.9, Ridge=True)
plot_mse_r2(mse_adagrad, mse_rmsprop, mse_adam, mse_none, mse_none_mom, r2_adagrad, r2_rmsprop, r2_adam, r2_none, r2_none_mom, title="GD with Ridge", savename="Gradient_descent_using_Ridge.pdf")
#print("Finished MSE and R2 plots")



batch_list = [2, 4, 8, 16, 32]

plot_batch_size(batch_list, scheduler="ada", learning_rate=0.3, Ridge=False, title="Adagrad")
plot_batch_size(batch_list, scheduler="rms", Ridge=False, title="Rmsprop")
plot_batch_size(batch_list, scheduler="adam", Ridge=False, title="Adam")
plot_batch_size(batch_list, scheduler=None, Ridge=False, title="None")
plot_batch_size(batch_list, momentum=0.9, scheduler=None, Ridge=False, title="None_with_momentum")
#print("Finished all plots")
