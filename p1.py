import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Generate random values for x and y within [0, 1]
np.random.seed(0)  # Setting a seed for reproducibility
n_points = 100  # Number of data points
x = np.random.rand(n_points)
y = np.random.rand(n_points)


# Generate the corresponding z values using the Franke function
z = FrankeFunction(x, y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.33, random_state=42)

# Step 2: Add random noise (N(0, 1)) to the Franke function values
noise = np.random.normal(0, 1, n_points)
z_with_noise = z + noise

# Create polynomial features up to fifth order
max_degree = 5
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(np.vstack((x, y)).T)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, z_with_noise)

# Make predictions using the model
predictions = model.predict(X_poly)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(z_with_noise, predictions)
#print("Mean Squared Error (MSE):", mse)


# Calculate R^2 score
r2 = r2_score(z_with_noise, predictions)

# Create an array to store MSE and R^2 values for different degrees
degrees = range(1, max_degree + 1)
mse_values = []
r2_values = []
beta_values = []

for degree in range(1, max_degree+1):
    X_train = np.column_stack([x_train**i * y_train**(degree-i) for i in range(degree+1)])
    X_test = np.column_stack([x_test**i * y_test**(degree-i) for i in range(degree+1)])

    # Center and scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform ordinary least squares regression
    beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

    # Make predictions
    z_pred = X_test_scaled @ beta

    # Calculate MSE and R-squared
    mse = mean_squared_error(z_test, z_pred)
    r2 = r2_score(z_test, z_pred)

    mse_values.append(mse)
    r2_values.append(r2)

# Plot MSE and R-squared scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, max_degree+1), mse_values, '.-')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(range(1, max_degree+1), r2_values, '.-')
plt.title('R-squared (R^2) Values')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2')

plt.tight_layout()
plt.show()

# Print the estimated beta coefficients for the highest degree
best_degree = np.argmax(r2_values) + 1
best_beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train
print(f"Best polynomial degree: {best_degree}")
print(f"Estimated beta coefficients for degree {best_degree}:")
for i, coeff in enumerate(best_beta):
    print(f"beta[{i}] = {coeff:.4f}")


"""
# Perform the analysis for different polynomial degrees
for d in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(np.vstack((x, y)).T)

    # Fit a linear regression model
    model.fit(X_poly, z_with_noise)

    # Make predictions
    predictions = model.predict(X_poly)

    # Calculate MSE and R^2
    mse = mean_squared_error(z_with_noise, predictions)
    r2 = r2_score(z_with_noise, predictions)

    mse_values.append(mse)
    r2_values.append(r2)

    # Store the coefficients (parameters) beta
    beta = model.coef_
    beta_values.append(beta)

# Plot MSE and R^2 as functions of the polynomial degree
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(degrees, mse_values, '.-')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Polynomial Degree')

plt.subplot(1, 3, 2)
plt.plot(degrees, r2_values,'.-')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.title('R^2 Score vs. Polynomial Degree')

# Plot parameters (beta) as you increase the order of the polynomial
plt.subplot(1, 3, 3)
for i in range(len(beta_values[0])):
    plt.plot(degrees, [beta[i] for beta in beta_values], '.-', label=f'β{i}')
plt.xlabel('Polynomial Degree')
plt.ylabel('Parameter Value (β)')
plt.title('Parameter Values vs. Polynomial Degree')
plt.legend()

plt.tight_layout()
plt.show()
"""
