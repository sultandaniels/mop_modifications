import numpy as np

# Example data
x_values = np.array([1, 2, 3, 4, 5])  # Independent variable
y_values = np.array([2e1, 2e0, 2e-1, 2e-2, 2e-3])  # Dependent variable

def loglogfit(x_values, y_values):
    # For a log-log scale regression
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    print("log_y:", log_y)

    # Set up the design matrix for a linear model
    A = np.vstack([log_x, np.ones(len(log_x))]).T

    # Solve the least squares problem
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    # To plot or use the regression line:
    # Convert back if you're working on a log-log scale
    predicted_y = np.exp(m * log_x + c)
    return predicted_y, m, c

predicted_y,m,c = loglogfit(x_values, y_values)

# Plot the data and the regression line
import matplotlib.pyplot as plt
plt.scatter(x_values, y_values, label="Data")
plt.plot(x_values, predicted_y, label="Regression Line", color="red")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()