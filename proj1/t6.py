from functions import *
plt.rcParams.update({"font.size": 15})

def lamda_degree_MSE(x, y, z, method, std, n_lmb = 50, maxdegree = 15, k_folds = 5, max_iter = 100, save=True, lmb_min=-12, lmb_max=-1):
    """
    Function to find best degree and lambda parameter
    for the chosen regression method
    takes in:
    - x:                meshrgrid containing x-values
    - y:                meshrgrid containing y-values
    - z:                data
    - method:           Regression method
    - std:              standard deviation of noise added to data
    - n_lmb (opt):      number of lambdas to test for logspace (lmb_min, lmb_max)
    - maxdegree (opt):  maximum degree to test the regression
    - k_folds (opt):    number of kfolds for cross validation method
    - max_iter (opt):   maximum number of iterations used for lasso prediction
    - save (opt):       if true saves ploted heatmap.
    - lmb_min (opt):    minimum power of 10 for lambda (10^(lmb_min))
    - lmb_max (opt):    maximum power of 10 for lambda  (10^(lmb_max))
    returns:
    - optimal lamda, degree and the mse for
    """

    degree = np.arange(1, maxdegree+1)
    lamda = np.logspace(lmb_min, lmb_max, n_lmb)

    if method == "RIDGE" or method == "LASSO":
        degree, lamda = np.meshgrid(degree,lamda)
        mse = np.zeros(np.shape(degree))

        for i in range(maxdegree):
            X = design_matrix(x, y, degree[0, i])
            for j in range(n_lmb):
                mse[j, i] = cross_validation(X, z, k_folds, lamda[j, i], method, max_iter)
            print("\n\n\n ---DEGREE---- %i\n\n\n" %(i))

    elif method == "OLS":
        mse = np.zeros(np.shape(degree))
        for i in range(maxdegree):
            X = design_matrix(x, y, degree[i])
            mse[i] = cross_validation(X, z, k_folds, method=method)

    argmin = np.unravel_index(np.argmin(mse), mse.shape)

    print("---%s---" %(method))
    print("Degree of lowest MSE for %i kfolds" %(k_folds), degree[argmin])
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if method != "OLS":
        print("Lambda of lowest MSE for %i kfolds" %(k_folds), lamda[argmin])
        plt.contourf(degree, lamda, mse, 50, cmap="RdBu")
        plt.colorbar(label=r"$MSE$")
        plt.ylabel(r"$\lambda$")
        plt.yscale("log")
        plt.scatter(degree[argmin], lamda[argmin], marker="x", s=80, label=r"min MSE: %.3f, Lambda: %.2e" %(mse[argmin], lamda[argmin]))
        plt.legend(fontsize=12)

    else:
        plt.plot(degree, mse, "--o", fillstyle="none")
        plt.ylabel(r"$MSE$")
        plt.scatter(degree[argmin], mse[argmin], color="k", marker="x", s=80, label="min MSE: %.3f" %(mse[argmin]))
        plt.legend()

    plt.xlabel("Degree")
    plt.grid(True)

    if save:
        plt.savefig("../figures/best_lambda_%s_0%i.png" %(method, std*10), dpi=300, bbox_inches='tight')
    plt.show()

    if method == "OLS":
        return lamda[0], degree[argmin], mse[argmin]
    else:
        return lamda[argmin], degree[argmin], mse[argmin]


def compare_3d(x, y, z, noise, deg_ols, lmb_ridge, deg_ridge, lmb_lasso, deg_lasso, name_add="franke", std=1, mean=0, azim=50):
    """
    Plots 3D surface for OLS, RIDGE and LASSO regression for the chosen degrees
    and lambdas. Saves the files giving a total of 6 plots.
    takes in:
    - x:            meshrgrid containing x-values
    - y:            meshrgrid containing y-values
    - z:            data with added noise
    - noise:        added noise of data
    - deg_...:      Degrees to plot for the different regressions
    - lmb_...:      Lambdas to use for plot for lasso and ridge
    - name_add:     string to add at end of saved filenames
    - std (opt):    std used to reduce standard scale of z
    - mean (opt):   mean used to reduce standard scale of
    - azim (opt):   azim in degrees for initial position of 3D plot

    plots:
    - 6 plots with surfaces.
    Test and train data, true data and, ridge, ols and lasso regressions
    """
    z_true = (z*std + mean  - noise)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z, test_size=0.2)

    X_train = design_matrix(x_train, y_train, deg_ridge)
    X_test = design_matrix(x_test, y_test, deg_ridge)
    beta = ridge_regression(X_train, z_train, lmb_ridge)
    z_pred_ridge = (X_test @ beta)*std + mean

    X_train = design_matrix(x_train, y_train, deg_lasso)
    X_test = design_matrix(x_test, y_test, deg_lasso)
    lasso = lasso_regression(X_train, z_train, lmb_lasso, max_iter=int(1e3), tol=1e-4)
    z_pred_lasso = lasso.predict(X_test)*std + mean

    X_train = design_matrix(x_train, y_train, deg_ols)
    X_test = design_matrix(x_test, y_test, deg_ols)
    beta = OLS(X_train, z_train)
    z_pred_OLS =  (X_test @ beta)*std + mean

    plot_3d_trisurf(x_test, y_test, z_test*std + mean , azim=azim, title="Test data")
    plt.savefig("../figures/test_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_train, y_train, z_train*std + mean , azim=azim, title="Train data")
    plt.savefig("../figures/train_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x.ravel(), y.ravel(), z_true, azim=azim, title="Actual data")
    plt.savefig("../figures/actual_data_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_ridge, azim=azim, title="Ridge predict")
    plt.savefig("../figures/ridge_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_lasso, azim=azim, title="Lasso predict")
    plt.savefig("../figures/lasso_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plot_3d_trisurf(x_test, y_test, z_pred_OLS, azim=azim, title="OLS predict")
    plt.savefig("../figures/ols_pred_%s.png" %(name_add), dpi=300, bbox_inches='tight')
    plt.show()

def compare_beta_lambda(x, y, z, lamda):
    """
    Function to plot how beta parameters react to different values of
    lamda for a design matrix of degree 5

    takes in:
    - x:                meshrgrid containing x-values
    - y:                meshrgrid containing y-values
    - z:                data
    - lamda :           numpy 1D array of lambdas to plot for
    """
    X = design_matrix(x, y, 5)
    beta_ridge = np.zeros((len(lamda), X.shape[1]))
    beta_lasso = np.zeros((len(lamda), X.shape[1]))
    i=0
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    for lmb in lamda:
        beta_ridge[i] = ridge_regression(X, z, lmb)
        beta_lasso[i] = lasso_regression(X_train, z_train, lmb, max_iter=10000).coef_
        i +=1

    plt.plot(lamda, beta_ridge)
    plt.title(r"Ridge $\beta$ for degree of 5")
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    plt.savefig("../figures/ridge_beta.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(lamda, beta_lasso)
    plt.title(r"Lasso $\beta$ for degree of 5")
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    plt.savefig("../figures/lasso_beta_test.png", dpi=300, bbox_inches='tight')
    plt.show()








n = 30
#n = 200
std = 0.2
maxdegree = 15
x, y, z = make_data(n, std, seed=200)
np.random.seed(200)
noise = np.random.normal(0, std, size=(n+1,n+1)).ravel()

run_best_lambda_plots = False #To run calculations for best degree and lamda (may take some time). if False uses already calculated values

if run_best_lambda_plots:
    lmb, deg, mse = lamda_degree_MSE(x, y, z, "OLS", std, save=True, maxdegree=15)
    lmb_ridge, deg_ridge, mse_ridge = lamda_degree_MSE(x, y, z, "RIDGE", std, save=True, maxdegree=15, lmb_min =-8, n_lmb=30)
    lmb_lasso, deg_lasso, mse_lasso = lamda_degree_MSE(x, y, z, "LASSO", std, save=True, maxdegree=20, lmb_min =-12, n_lmb=30, max_iter=1000)

    print("OLS:", lmb, deg, mse)
    print("RIDGE:", lmb_ridge, deg_ridge, mse_ridge)
    print("LASSO:", lmb_lasso, deg_lasso, mse_lasso)

else:
    lmb_ridge = 1.6102620275609392e-07
    deg_ridge = 8
    mse_ridge = 0.03902354229755195
    lmb_lasso = 1.0826367338740564e-09
    deg_lasso = 19
    mse_lasso = 0.04242593103561867
    deg_ols = 6
    mse_ols = 0.03930695369808278

"""3D plot of the predictions"""
compare_3d(x, y, z, noise, deg_ols, lmb_ridge, deg_ridge, lmb_lasso, deg_lasso, name_add="franke_extra")
