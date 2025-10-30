import matplotlib.pyplot as plt
from math import cos, sin, degrees

# Target position (x*, y*) to reach.
target = [-0.8, 1.2]

# Lengths of the three arm links l1, l2, l3.
l1 = l2 = l3 = 1.0

# Used for central difference first derivative
h = 1e-5

# Initial guess for the angles [θ1, θ2, θ3] in radians.
theta0 = [0.2,0.1,-0.3]

# When the Euclidean distance between the current end position and the target\
# falls below this value, we declare convergence.
max_distance = 0.01

# Learning rate (step size) for GD.
alpha = 0.01

#
def P(angles):
    x = l1 * cos(angles[0]) + l2 * cos(angles[0] + angles[1]) + l3 * cos(angles[0] + angles[1] + angles[2])
    y = l1 * sin(angles[0]) + l2 * sin(angles[0] + angles[1]) + l3 * sin(angles[0] + angles[1] + angles[2])
    return [x, y]

def two_norm_squared(vec):
    """
    Squared Euclidean norm ||v||^2 of a 2D vector [vx, vy] (or any iterable of numbers).
    Using the squared norm avoids a sqrt and is sufficient for L(θ) and gradient computations.
    """
    return sum(i**2 for i in vec)

def L(angles):
    """
    Loss function: L(θ) = 0.5 * ||P(θ) - target||^2
    """
    p_theta = P(angles)
    return 0.5 * two_norm_squared([p_theta[0] - target[0], p_theta[1] - target[1]])

def nablaL(angles):
    """
    Numerical first derivate by central difference.
    """
    grad = [0, 0, 0]
    for i in range(3):
        # Create perturbed copies of the angle vector for central difference.
        angles_plus_h = angles[:]
        angles_minus_h = angles[:]
        
        angles_plus_h[i] += h
        angles_minus_h[i] -= h

        # Central-difference approximation to the partial derivative along dimension i.
        grad[i] = (L(angles_plus_h) - L(angles_minus_h)) / (2 * h)
    return grad

def calculate_euclidean_distance(angles):
    """
    Euclidean distance = ||P(θ) - target||
    """
    p_theta = P(angles)
    return (two_norm_squared([p_theta[0] - target[0], p_theta[1] - target[1]]))**0.5

# -------------------------------------------------------------------------------------------------
# Gradient Descent (GD)
print("---------- Gradient Descent ----------")

# Initialize iteration counter and starting configuration.
k = 0
angles_k = theta0[:]           # copy to avoid mutating the original initial guess
path_gd = [P(angles_k)]        
dist = calculate_euclidean_distance(angles_k)


while (dist > max_distance):
    grad = nablaL(angles_k)                                      # compute ∇L(θ_k)
    angles_k = [angles_k[i] - alpha * grad[i] for i in range(3)] # gradient step
    path_gd.append(P(angles_k))                                  # record path (optional)
    dist = calculate_euclidean_distance(angles_k)                # update convergence metric
    k += 1

print(f"Converged in {k} iterations.")
print("Final Angles (deg): [" + ", ".join(f"{degrees(a):.4f}" for a in angles_k) + "]")
print("Final Co-ordinates: [" + ", ".join(f"{v:.4f}" for v in P(angles_k)) + "]")
print(f"Final Distance: {dist:.5f}\n")

# -------------------------------------------------------------------------------------------------
# Gradient Descent with Momentum (GDM)
print("---------- Gradient Descent with Momentum ----------")

# Momentum coefficient and initial velocity vector.
beta = 0.9
v_k = [0.0, 0.0, 0.0]

# Reset the iteration counter, configuration, and distance for the momentum run.
k = 0
angles_k = theta0[:]                                       # start again from the same initial guess
path_gdm = [P(angles_k)]
dist = calculate_euclidean_distance(angles_k)              # compute initial distance

while (dist > max_distance):
    grad = nablaL(angles_k)                                    # compute current gradient
    v_k = [beta * v_k[i] + alpha * grad[i] for i in range(3)] # update velocity
    angles_k = [angles_k[i] - v_k[i] for i in range(3)]       # update parameters
    path_gdm.append(P(angles_k))                               # record path (optional)
    dist = calculate_euclidean_distance(angles_k)              # update convergence metric
    k += 1

print(f"Converged in {k} iterations.")
print("Final Angles (deg): [" + ", ".join(f"{degrees(a):.4f}" for a in angles_k) + "]")
print("Final Co-ordinates: [" + ", ".join(f"{v:.4f}" for v in P(angles_k)) + "]")
print(f"Final Distance: {dist:.5f}")

# -------------------------------------------------------------------------------------------------
# Visualization
print("\nGenerating plot...")

# Create a figure and axes for the plot.
fig, ax = plt.subplots(figsize=(16, 9))

# Unzip the path coordinates for plotting.
x_gd, y_gd = zip(*path_gd)
x_gdm, y_gdm = zip(*path_gdm)

# Plot the path for Gradient Descent.
ax.plot(x_gd, y_gd, 'o-', color='dodgerblue', markersize=3, linewidth=1.5, label='GD Path')

# Plot the path for Gradient Descent with Momentum.
ax.plot(x_gdm, y_gdm, 'o-', color='orangered', markersize=3, linewidth=1.5, label='GDM Path')

# Mark the key points on the plot for clarity.
# Start Position (common to both)
ax.plot(path_gd[0][0], path_gd[0][1], 's', color='black', markersize=10, label='Start')
# Target Position
ax.plot(target[0], target[1], 'X', color='limegreen', markersize=15, label='Target')

# Add labels, title, legend, and grid for a well-labeled plot.
ax.grid(True)
ax.legend()
ax.set_aspect('equal', adjustable='box') # Ensures x and y axes have the same scale.

# Display the plot.
plt.show()