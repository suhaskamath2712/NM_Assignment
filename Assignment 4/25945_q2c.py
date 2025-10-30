import math
import matplotlib.pyplot as plt

# --- Helper function for the Logistic Map ---
def logistic_map(A, x0, n_iter):
    """
    Generate iterations of the logistic map: x_{n+1} = A*x_n*(1 - x_n).

    Args:
        A (float): Parameter A.
        x0 (float): Initial condition (0 < x0 < 1).
        n_iter (int): Total number of iterations to perform.

    Returns:
        list[float]: List of logistic map values [x_0, x_1, ..., x_{n_iter-1}].
    """
    # Initialize list with n_iter elements (list of zeros)
    x = [0.0] * n_iter
    x[0] = x0
    for i in range(1, n_iter):
        x[i] = A * x[i-1] * (1 - x[i-1])
    return x

# --- Helper function for generating range with fixed step ---
def custom_arange(start, stop, step):
    """
    Custom implementation of numpy.arange returning a standard list.
    Generates values from start to stop (inclusive of stop, accounting for floating point).
    """
    values = []
    current = start
    # Check if current is less than stop + a small epsilon to include the endpoint
    while current <= stop + step/2: 
        values.append(current)
        current += step
    return values

# --- Helper function to calculate the mean of a list ---
def custom_mean(data):
    """
    Calculates the mean of a list of numbers.
    """
    if not data:
        return 0
    return sum(data) / len(data)

# PART 1: Bifurcation Diagram

print("Generating Bifurcation Diagram...")

min_A, max_A = 0.89, 3.995
step = 0.0125
n_iter_bifurcation = 200
transients = 15

# 1. Use custom_arange for robust generation of A values
A_values = custom_arange(min_A, max_A, step)

# Arrays to collect all A and x values for plotting
all_A = []
all_x = []

for A in A_values:
    # 1. Generate the sequence
    x_sequence = logistic_map(A, x0=0.5, n_iter=n_iter_bifurcation)
    
    # 2. Discard transients and keep only the final steady states
    steady_state_x = x_sequence[transients:]
    
    # 3. Store A (repeated) and x values
    all_A.extend([A] * len(steady_state_x))
    all_x.extend(steady_state_x)

# Plotting the Bifurcation Diagram
plt.figure(figsize=(16,9))
# Use plt.plot with minimal marker size and low alpha for density visualization
plt.plot(all_A, all_x, '.', alpha=0.7)

plt.title("Bifurcation Diagram of the Logistic Map")
plt.xlabel("Control Parameter A")
plt.ylabel("$x_n$ (Steady-State Values)")
plt.xlim(min_A, max_A)
plt.ylim(0, 1) #x can only be between 0 and 1
plt.savefig(r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q2c_bifurcation.png", dpi=300)

# PART 2: Lyapunov Exponent and Sensitivity to Initial Conditions

print("Calculating Lyapunov Exponent...")

A_lyapunov = 4.0
x0 = 0.5            # Initial condition
delta0 = 0.01       # Initial separation
x0_prime = x0 + delta0 # Second initial condition
n_iter_lyapunov = 500  # More iterations for a better estimate

# 1. Generate both sequences
x_seq = logistic_map(A_lyapunov, x0, n_iter_lyapunov)
x_prime_seq = logistic_map(A_lyapunov, x0_prime, n_iter_lyapunov)

# 2. Calculate the separation at each step, |x_n - x'_n| and the exponent lambda_n
n_steps = []
lambda_n = []

# Loop starts at n=1 (index 1) since the formula is defined for n > 0
for n in range(1, n_iter_lyapunov):
    # Separation at time n
    delta_n = abs(x_seq[n] - x_prime_seq[n])
    
    # Calculate the finite-time Lyapunov exponent
    # Formula: lambda_n = (1/n) * log(delta_n / delta_0)
    # Use math.log (natural logarithm)
    # Added a small check to prevent log(0) errors if sequences converge or collide
    if delta_n > 1e-15:
        current_lambda = (1.0 / n) * math.log(delta_n / delta0)
        lambda_n.append(current_lambda)
        n_steps.append(n)
    else:
        # If delta_n is effectively zero, lambda is negative infinity, which is skipped for the plot
        pass 
        

# Plotting the Lyapunov Exponent
plt.figure(figsize=(16, 9))
plt.plot(n_steps, lambda_n, 'b-', label=r'Finite-Time Lyapunov Exponent $\lambda_n$')

# Calculate the mean (asymptotic) value for large n
# We'll take the mean of the last 100 values as the estimate
# Ensure lambda_n has enough points, otherwise take the whole list
asymptotic_lambda_data = lambda_n[-100:] if len(lambda_n) > 100 else lambda_n
asymptotic_lambda = custom_mean(asymptotic_lambda_data)

# To plot the horizontal line, we need the start and end point for the line
plt.axhline(asymptotic_lambda, color='r', linestyle='--', 
            label=f'Asymptotic Value $\\approx$ {asymptotic_lambda:.4f}')

plt.title(r"Lyapunov Exponent $\lambda_n$ for Logistic Map ($A=4$)")
plt.xlabel("Iteration Count (n)")
plt.ylabel(r"$\lambda_n$")
plt.xlim(0, n_iter_lyapunov)
plt.legend()
plt.tight_layout()
plt.savefig(r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q2c_lyapunov.png", dpi=300)

print(f"\nReported Asymptotic Lyapunov Exponent for A={A_lyapunov}: {asymptotic_lambda:.4f}")
