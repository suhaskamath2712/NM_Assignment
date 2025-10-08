import matplotlib.pyplot as plt
import numpy as np

#Function
def f(x):  
    return (x**3) + (2 - 0.5j)*(x**2) + (1.5 + 1j)*x + 0.2j

def fprime(x):
    return 3*x**2 + (4 - 1j)*x + (1.5 + 1j)

#Relative error function for complex numbers
def rel_error(x_new:complex, x_old:complex):
    #Magnitude of new value
    mag_x_new = abs(x_new)

    #Handle case where new value is zero to avoid division by zero
    if mag_x_new == 0:
        return 0.0 if x_old == 0 else float('inf')
    
    #Magnitude of the difference between the two numbers (absolute error)
    abs_error = abs(x_new - x_old)
    return abs_error / mag_x_new

n = 0   #Iteration counter
epsilon  = 1e-6  #Tolerance

#Initial guesses
x0, x1, x2 = 0 + 0j, 1 + 0.5j, 2 + 1j
tolerance = 1e-6
max_iterations = 20
iteration = 0
errors = []

print("--- Muller's Method for Complex Root Finding ---")
print(f"Initial guesses: x0={x0}, x1={x1}, x2={x2}")
print(f"Tolerance: {tolerance}\n")

#Start iterating
for iteration in range(1, max_iterations + 1):
    # Evaluate function at current points
    f0, f1, f2 = f(x0), f(x1), f(x2)

    # Calculate divided differences for the parabola coefficients
    d1 = (f1 - f0) / (x1 - x0)
    d2 = (f2 - f1) / (x2 - x1)
    
    # Coefficients of the parabola: a(x-w2)² + b(x-w2) + c
    a = (d2 - d1) / (x2 - x0)
    b = d2 + a * (x2 - x1)
    c = f2
    
    # Calculate the denominator, choosing the sign to maximize its magnitude for stability
    discriminant = (b**2 - 4*a*c)**0.5
    if abs(b + discriminant) > abs(b - discriminant):
        denominator = b + discriminant
    else:
        denominator = b - discriminant
    
    # Calculate the next approximation for the root
    delta_x = -2 * c / denominator
    x_next = x2 + delta_x
    
    # Calculate relative error
    if abs(x_next) == 0:
        relative_error = abs(delta_x)
    else:
        relative_error = abs(delta_x) / abs(x_next)
    
    errors.append(relative_error)
    
    print(f"Iteration {iteration}: x = {x_next:.6f}, Relative Error = {relative_error:.2e}")
    
    # Check for convergence
    if relative_error < tolerance:
        break
        
    # Update points for the next iteration
    x0, x1, x2 = x1, x2, x_next

# Report the results
print(f"\nConverged after {iteration} iterations.")
print(f"Approximate root: {x_next}")
print(f"Function value at root: {f(x_next)}")
print(f"Final relative error: {relative_error}")

plt.clf()
plt.plot(np.log(errors[:-1]), np.log(errors[1:]), marker='o', label='Relative Error vs. Iterations')
y = [1.84*i for i in np.log(errors[:-1])]
plt.plot(np.log(errors[:-1]), y, label='Reference Line: y = 1.84x')
plt.xlabel('log(e_n)')
plt.ylabel('log(e_n+1)')
plt.legend()
plt.grid(True)
plt.show()

print("----------------------------------------------------------------")

print("Trying to solve with Newton's method:")

n = 0   #Iteration counter
epsilon = 1e-6  #Tolerance level

#Initial guess
x0 = 2 + 1j
print("Initial guess:",f"x = {x0} \n")

x1 = x0 - f(x0)/fprime(x0)
error_margin = rel_error(x1, x0)

while error_margin > epsilon:
    #Print current iteration details
    print(f"Iteration {n}: x = {x1}, f(x) = {f(x1)}, Relative Error = {error_margin}")

    #Compute new guess using Newton's method formula
    x0 = x1
    x1 = x0 - f(x0)/fprime(x0)

    error_margin = rel_error(x1,x0)
    n += 1

print(f"\nConverged after {n} iterations.")
print(f"Approximate root: {x1}")
print(f"Function value at root: {f(x1)}")
print(f"Final relative error: {error_margin}")
