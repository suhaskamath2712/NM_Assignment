import matplotlib.pyplot as plt
import numpy as np

def getBestFit(x, y, degree):
    num_points = len(x)

    # Construct Vandermonde matrix
    A = np.array([[x[i]**j for j in range(degree + 1)] for i in range(num_points)])

    # Solve for coefficients using the normal equation
    coeffs = np.linalg.solve(A.T @ A, A.T @ y)
    return coeffs[::-1]


def evaluatePolynomial(coeffs, x):
    degree = len(coeffs) - 1
    return sum(c * (x ** (degree - i)) for i, c in enumerate(coeffs))

print("Question 4a")
x = [0, 1, 2, 3, 4]
y = [1, 2.2, 3.9, 6.1, 8.0]
deg = 2
coefficients = getBestFit(x, y, deg)
print("Best fit polynomial coefficients (descending order):", coefficients)

degree_print = len(coefficients) - 1
print("Best fit polynomial: P(x) = " + " + ".join(f"{coeff:.4f}x^{degree_print - i}" for i, coeff in enumerate(coefficients)).replace("x^1", "x").replace("x^0", ""))

y_pred = [evaluatePolynomial(coefficients, xi) for xi in x]
mse = sum([(y_act - y_pre)**2 for y_act, y_pre in zip(y, y_pred)]) / len(y)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# Generate more points for a smooth curve between min and max of x
x_smooth = [min(x) + (max(x) - min(x)) * i / 99.0 for i in range(100)]
y_smooth = [evaluatePolynomial(coefficients, xi) for xi in x_smooth]

# Plotting the data points and the best fit curve
plt.plot(x, y, 'o', label='Original Data points')
plt.plot(x_smooth, y_smooth, label='Best Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset (a): Degree 2 Polynomial Fit')
plt.legend()
plt.grid()
plt.show()

print("========================================================================")
print("Question 4b")

x = [-3, -2, -1, 0, 1, 2, 3]
y = [-2.5, -1, 0.5, 0, 0.6, 2.5, 1]
deg = 4

coefficients = getBestFit(x, y, deg)
print("Best fit polynomial coefficients (descending order):", coefficients)

degree_print = len(coefficients) - 1
print("Best fit polynomial: P(x) = " + " + ".join(f"{coeff:.4f}x^{degree_print - i}" for i, coeff in enumerate(coefficients)).replace("x^1", "x").replace("x^0", ""))

y_pred = [evaluatePolynomial(coefficients, xi) for xi in x]
mse = sum([(y_act - y_pre)**2 for y_act, y_pre in zip(y, y_pred)]) / len(y)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# Generate more points for a smooth curve between min and max of x
x_smooth = [min(x) + (max(x) - min(x)) * i / 99.0 for i in range(100)]
y_smooth = [evaluatePolynomial(coefficients, xi) for xi in x_smooth]

# Plotting the data points and the best fit curve
plt.plot(x, y, 'o', label='Original Data points')
plt.plot(x_smooth, y_smooth, label='Best Fit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset (b): Degree 4 Polynomial Fit')
plt.legend()
plt.grid()
plt.show()

print("========================================================================")