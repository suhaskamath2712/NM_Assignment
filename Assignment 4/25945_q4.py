import sys
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D # For custom legend
from matplotlib.patches import Circle  # for drawing epicycle circles

def compute_fourier_coefficients(z_samples, M):
    """
    Computes the Fourier coefficients c_k for k from -M to M using a
    numerical approximation of the continuous integral (Equation 4).
    """

    # N is the total number of discrete samples
    N = len(z_samples)
    if N == 0:
        print("Error: Input array z_samples is empty.")
        return {}

    # This dictionary will store our coefficients, mapping k -> c_k
    coefficients = {}

    # Loop through all required mode numbers k, from -M to M (inclusive)
    for k in range(-M, M + 1):

        # Initialize the sum for this specific coefficient k
        # We start with a complex number 0+0j
        sum_k = 0.0 + 0.0j

        # Now, loop through all N samples (j from 0 to N-1)
        # to compute the sum part of the formula
        for j in range(N):
            # Get the j-th complex data point
            z_j = z_samples[j]

            # Calculate the exponential term: -i * 2 * pi * k * j / N
            # In Python and numpy, 1j is used for the imaginary unit i
            exponent = -1j * 2 * np.pi * k * j / N

            # Add this term to the sum: z_j * exp(exponent)
            sum_k += z_j * np.exp(exponent)

        # After summing over all j, we apply the (1/N) scaling
        # to get the final coefficient c_k
        c_k = sum_k / N

        # Store the computed coefficient in our dictionary
        coefficients[k] = c_k

    # Return the complete dictionary of coefficients
    return coefficients

def reconstruct_curve(coefficients, M, N_recon=1000):
    """
    Reconstructs the letter outline from a set of Fourier coefficients
    using the synthesis Equation 5.
    """

    # Create the time array for reconstruction:
    # N_recon equally spaced points from t=0 up to (but not including) t=1.
    t_recon = np.linspace(0, 1, N_recon, endpoint=False)

    # Initialize the reconstructed curve array with zeros
    # We will add the contribution of each mode (k) to this sum
    z_recon = np.zeros(N_recon, dtype=complex)

    # Loop through each mode from -M to M
    for k in range(-M, M + 1):
        # Get the coefficient c_k
        c_k = coefficients.get(k, 0.0 + 0.0j)

        # Calculate the exponential term for all t points at once (vectorized)
        # exp(i * 2 * pi * k * t)
        exponent_term = np.exp(1j * 2 * np.pi * k * t_recon)

        # Add this mode's contribution to the total sum
        z_recon += c_k * exponent_term

    # Return the time array and the reconstructed curve
    return t_recon, z_recon

# --- Part 2: Data Loading and Preprocessing ---
pts = np.loadtxt(r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\S.csv", delimiter=',', skiprows=1)

# Construct the complex values
points = pts[:,0] + 1j*pts[:,1]
z_samples_original = points # Use a name consistent with assignment

# Construct an array with parameter values
N = len(points)
t = [j/N for j in range(N)]

print(f"Loaded {N} points from 'S.csv'.")

# --- Plot original loaded letter ---
plt.figure(figsize=(4, 4))
plt.plot(points.real, points.imag, '-', label="Original Letter points")
plt.axis('equal')
plt.title("Parametric points of letter 'S' (from CSV)")
plt.legend()
plt.show()

# --- Part 5: Visualization ---
print("\n--- Starting Part 5: Plotting Subplots ---")
# Define parameters
N_reconstruction_points = 1000 
M_values = [2, 8, 32, 64, 128]

# --- Pre-compute all necessary Fourier coefficients ---
max_M_for_coeffs = max(M_values)
print(f"Computing Fourier coefficients up to M={max_M_for_coeffs}...")
all_coeffs = compute_fourier_coefficients(z_samples_original, max_M_for_coeffs)
print("Coefficients computed.")

# --- Plotting Subplots ---
fig, axes = plt.subplots(1, len(M_values), figsize=(15, 5))
fig.suptitle('Fourier Reconstruction')

x_original = z_samples_original.real
y_original = z_samples_original.imag

all_x = [x_original.min(), x_original.max()]
all_y = [y_original.min(), y_original.max()]

for i, M in enumerate(M_values):
    print(f"Reconstructing curve with M={M} modes...")
    t_recon, z_recon = reconstruct_curve(all_coeffs, M, N_reconstruction_points)

    x_recon = z_recon.real
    y_recon = z_recon.imag
    
    all_x.extend([x_recon.min(), x_recon.max()])
    all_y.extend([y_recon.min(), y_recon.max()])

    ax = axes[i]
    ax.plot(x_original, y_original, color='gray', linestyle='-', linewidth=1, label='Original')
    ax.plot(x_recon, y_recon, color='red', linestyle='-', linewidth=2, label='Reconstructed')
    
    ax.set_title(f'mode-{M}')
    ax.set_aspect('equal', adjustable='box') 
    ax.axis('off') 

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

output_filename_plot = r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q4_5.png"
plt.savefig(output_filename_plot, dpi=300)
print(f"Part 5 plot saved to {output_filename_plot}")

# --- Part 6: Animation ---
print("\n--- Starting Part 6: Animation (Epicycle-only style) ---")
# --- Parameters ---
M_to_animate = 64         # Number of modes for animation
N_frames = 400            # Number of frames in the animation

# --- Prepare Data for Animation ---
# Use coefficients already computed up to max M
print(f"Preparing animation data for M={M_to_animate}...")
coeffs_sorted = []
coeffs_sorted.append((0, all_coeffs.get(0, 0j)))  # DC term (center shift)
for k in range(1, M_to_animate + 1):
    coeffs_sorted.append((k, all_coeffs.get(k, 0j)))
    coeffs_sorted.append((-k, all_coeffs.get(-k, 0j)))

t_anim = np.linspace(0, 1, N_frames, endpoint=False)
num_vectors = len(coeffs_sorted)

# Precompute centers/endpoints of each epicycle chain for every frame
# Shape: [frame, vector_index(0..num_vectors), (x,y)]
# column 0 is the origin (center of first circle), column j is the center for vector j,
# column j+1 is the endpoint after adding vector j.
epicycles_data = np.zeros((N_frames, num_vectors + 1, 2), dtype=float)

for i, t_val in enumerate(t_anim):
    current_pos = 0j
    epicycles_data[i, 0, :] = [0.0, 0.0]  # start center at origin
    for j, (k, c_k) in enumerate(coeffs_sorted):
        vector = c_k * np.exp(1j * 2 * np.pi * k * t_val)
        next_pos = current_pos + vector
        # center for this vector:
        epicycles_data[i, j, :] = [current_pos.real, current_pos.imag]
        # endpoint after adding this vector:
        epicycles_data[i, j + 1, :] = [next_pos.real, next_pos.imag]
        current_pos = next_pos

# Path traced by the pen (last endpoint over time)
path_x = epicycles_data[:, -1, 0]
path_y = epicycles_data[:, -1, 1]

# --- Set up the plot ---
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

# Optional: plot the original letter as a faint reference
ax_anim.plot(z_samples_original.real, z_samples_original.imag, color='gray', alpha=0.25, linewidth=1)

# Axis limits from all points the mechanism visits
min_x_anim = epicycles_data[..., 0].min()
max_x_anim = epicycles_data[..., 0].max()
min_y_anim = epicycles_data[..., 1].min()
max_y_anim = epicycles_data[..., 1].max()
buffer_x_anim = (max_x_anim - min_x_anim) * 0.1
buffer_y_anim = (max_y_anim - min_y_anim) * 0.1

ax_anim.set_xlim(min_x_anim - buffer_x_anim, max_x_anim + buffer_x_anim)
ax_anim.set_ylim(min_y_anim - buffer_y_anim, max_y_anim + buffer_y_anim)
ax_anim.set_aspect('equal', adjustable='box')
ax_anim.set_title(f'Fourier Reconstruction (M={M_to_animate})')
ax_anim.axis('off')

# --- Create artists (circles, radial vectors, pen trace, pen point) ---
# Circle radius per vector (DC term has radius 0; it’s just a center shift)
radii = [0.0 if k == 0 else np.abs(c_k) for k, c_k in coeffs_sorted]

# Circles (one per vector)
circle_patches = []
for r in radii:
    circ = Circle((0.0, 0.0), r, edgecolor='tab:blue', facecolor='none', alpha=0.25, linewidth=1.0)
    ax_anim.add_patch(circ)
    circle_patches.append(circ)

# Radial vectors (lines from circle center to its rotating endpoint)
epicycle_lines = []
for _ in range(num_vectors):
    line, = ax_anim.plot([], [], color='tab:blue', alpha=0.6, linewidth=1.5)
    epicycle_lines.append(line)

# Optional small points at endpoints of each vector
epicycle_points, = ax_anim.plot([], [], 'o', color='tab:blue', alpha=0.5, markersize=2)

# Red pen and its trace (full path traced so far)
pen_trace, = ax_anim.plot([], [], color='red', linewidth=2)
pen_point, = ax_anim.plot([], [], 'o', color='red', markersize=4)

# --- Animation function ---
def update(frame):
    vec_pts = epicycles_data[frame]  # shape (num_vectors+1, 2)
    # Update circles (centers only; radii fixed)
    for j, circ in enumerate(circle_patches):
        cx, cy = vec_pts[j, 0], vec_pts[j, 1]  # center for vector j
        circ.center = (cx, cy)

    # Update radial vectors
    for j, line in enumerate(epicycle_lines):
        cx, cy = vec_pts[j, 0], vec_pts[j, 1]          # center
        ex, ey = vec_pts[j + 1, 0], vec_pts[j + 1, 1]  # endpoint
        line.set_data([cx, ex], [cy, ey])

    # Update small endpoints cloud (excluding the very first center)
    epicycle_points.set_data(vec_pts[1:, 0], vec_pts[1:, 1])

    # Update pen trace and pen point
    pen_trace.set_data(path_x[:frame + 1], path_y[:frame + 1])
    pen_point.set_data([path_x[frame]], [path_y[frame]])

    return (*circle_patches, *epicycle_lines, epicycle_points, pen_trace, pen_point)

# --- Create and save the animation ---
ani = FuncAnimation(fig_anim, update, frames=N_frames, interval=20, blit=True)

output_filename_anim = r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q4_6.gif"
print(f"Saving animation to {output_filename_anim}... (This may take a moment)")
ani.save(output_filename_anim, writer='pillow', fps=30)