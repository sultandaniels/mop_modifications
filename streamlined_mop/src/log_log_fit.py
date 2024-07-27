import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

def model_function(x, a, b, c):
    return c + np.exp(b) * x**a

# Define the loss function with regularization on b
def loss(lambda_reg, x_values, y_values, params):
    a, b, c = params
    y_pred = model_function(x_values, a, b, c)
    # Regular least squares loss
    loss_value = np.sum((y_values - y_pred)**2)
    # Add regularization term for b
    loss_value += lambda_reg * b**2
    return loss_value

def loglogfit(x_values, y_values):
    # Initial guess for the parameters [a, b, c]
    initial_guess = [-1.0, 0.0, 1.0]
    
    # Use curve_fit to fit the model function to the data
    params, covariance = curve_fit(model_function, x_values, y_values, p0=initial_guess)
    
    # Extract the parameters
    a, b, c = params
    
    # Generate y-values based on the fitted model
    fitted_y_values = model_function(x_values, a, b, c)

    print("Fitted parameters: a=%g, b=%g, c=%g" % (a, b, c))
    
    return fitted_y_values, a, b, c

def loglogfit_linear(x_values, y_values):
    # For a log-log scale regression
    log_x = np.log(x_values)
    log_y = np.log(y_values)

    # Set up the design matrix for a linear model
    A = np.vstack([log_x, np.ones(len(log_x))]).T

    # Solve the least squares problem
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    # To plot or use the regression line:
    # Convert back if you're working on a log-log scale
    predicted_y = np.exp(m * log_x + c)
    return predicted_y, m, c

def loglogfit_regularized(initial_guess, x_values, y_values, lambda_reg=0.01):
    ## regularized version
    # Initial guess for parameters

    # Perform the minimization
    result = minimize(lambda params: loss(lambda_reg, x_values, y_values, params), initial_guess)

    # Extract the optimized parameters
    a_opt, b_opt, c_opt = result.x
    return a_opt, b_opt, c_opt



if __name__ == '__main__':
    
    # Define the parameters for the curve
    a = -2.0  # Example value for a
    b = -0.5 # Example value for b
    c = 10   # Example value for c

    # Define the range of x values
    x_values = np.arange(1, 11)  # Generate integers from 1 to 10
    x_values = x_values.astype(float)  # Convert to float

    # Calculate y values using the curve equation
    y_values = c + np.exp(b) * x_values**a + np.random.normal(0, 1e-3, len(x_values))

    # Collect the ordered pairs
    ordered_pairs = list(zip(x_values, y_values))

    fitted_y_values, a_f, b_f, c_f = loglogfit(x_values, y_values)
    
    # Plot the data and the regression line
    plt.scatter(x_values, y_values-c_f, label="Data")
    plt.plot(x_values, fitted_y_values-c_f, label="Fitted Curve", color="red")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("y")

    lambda_reg = 0.01
    initial_guess = [-1.0, 0.0, 1.0]
    a_opt, b_opt, c_opt = loglogfit_regularized(initial_guess, x_values, y_values, lambda_reg)

    
    print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}")
    # Generate y-values based on the optimized model
    fitted_y_values_opt = model_function(x_values, a_opt, b_opt, c_opt)
    plt.plot(x_values, fitted_y_values_opt-c_opt, label="Regularized Fitted Curve", color="green")
    plt.legend()

    plt.show()