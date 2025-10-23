"""
Newton fractal for the polynomial f(x) = x^3 + 1

This script samples a rectangular region of the complex plane and applies
Newton's method to each complex seed. The basin-of-attraction (which root
each seed converges to) is recorded and rendered with matplotlib.

Key concepts and choices:
- The polynomial has three roots (one real, two complex conjugates). Newton's
  method will typically converge to the nearest attracting root depending on
  the seed. The boundaries between basins form the fractal.
- We record integer indices (0,1,2) for seeds that converge to one of the
  known roots within 'epsilon'. Non-convergent seeds (within max_iter) are
  left as their final complex iterate; imshow will map those values to colors
  as well (typically non-integer).
- This implementation uses pure Python lists (no numpy) for portability/constraints.
  It is straightforward but slower than a vectorized numpy/numba approach.
- Visualization: imshow maps the 2D list-of-lists to an image. Use 'extent'
  to map array indices to complex-plane coordinates. To have y increase upwards
  (mathematical convention), pass origin='lower' to imshow.
- Performance: complexity is O(width * height * avg_iters). On large grids this
  can take a long time in Python. For exploration reduce width/height or max_iter.
"""

import math
from matplotlib import pyplot as plt


# Problem-specific constants

# Precompute the three exact roots of x^3 + 1 = 0.
# These are:
#   1) -1 (real)
#   2) exp(i*pi/3)  = cos(pi/3) + i sin(pi/3)
#   3) exp(-i*pi/3) = cos(-pi/3) + i sin(-pi/3)
# Using trig functions is clear and numerically stable for these angles.
ROOTS = [
    -1,
    math.cos(math.pi / 3) + 1j * math.sin(math.pi / 3),
    math.cos(-math.pi / 3) + 1j * math.sin(-math.pi / 3)
]

# Convergence threshold: if iterate is within this Euclidean distance of a
# known root we consider the seed to have converged to that root.
epsilon = 1e-6

# Maximum number of Newton iterations to attempt for each seed before
# declaring the seed non-convergent (or at least not converged within budget).
max_iter = 100

# Function and derivative for Newton's method
def f(x):
    """
    Evaluate the polynomial f(x) = x^3 + 1 at complex x.
    """
    return x**3 + 1

def f_prime(x):
    """
    Derivative f'(x) = 3*x^2 used in the Newton step.
    """
    return 3 * x**2

def get_min_index(arr):
    """
    Return the minimum value in 'arr' and its index.

    Args:
        arr (iterable[float]): A small list of non-negative distances.

    Returns:
        (min_value, min_index)
    """
    # Simple linear scan; arrays are very short (length 3 in this script).
    min_index = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[min_index]:
            min_index = i
    return arr[min_index], min_index

def newton_fractal(x0):
    """
    Apply Newton's method to initial guess x0.

    Iteratively update x <- x - f(x)/f'(x) up to max_iter times. After every
    update compute distances to the known roots. If the current iterate is
    within 'epsilon' of a root, return the root index (0,1,2). If no root is
    reached within max_iter, returns -1.
    """
    x = x0
    for _ in range(max_iter):
        # Compute next Newton iterate. If f_prime(x) is zero (or extremely small)
        # this may produce a large step; robust code could check the magnitude and
        # break/perturb the iterate, but it's omitted here for clarity and brevity.
        x = x - f(x) / f_prime(x)

        # Compute distances to each known root and determine the nearest.
        distances = [abs(x - r) for r in ROOTS]
        min_dist, min_idx = get_min_index(distances)

        # If we are sufficiently close to a root, report the root index.
        if min_dist < epsilon:
            return min_idx

    # Not converged within max_iter: return the current iterate (complex).
    return x

def linspace(start, stop, num):
    """
    Minimal implementation of numpy.linspace with endpoint inclusion.

    Returns a list of length 'num' with values evenly spaced from start to stop,
    inclusive.
    """
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]

def zeros(shape):
    """
    Create a 1D or 2D Python list-of-lists containing zeros.

    Args:
        shape: tuple describing the desired shape. Supported:
               (n,) -> 1D list length n
               (rows, cols) -> 2D list rows x cols

    Returns:
        list or list[list]
    """
    if len(shape) == 1:
        return [0] * shape[0]
    elif len(shape) == 2:
        # Use a list comprehension to avoid shared-row aliasing.
        return [[0] * shape[1] for _ in range(shape[0])]
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")


# Grid and sampling parameters
# Image resolution: increase to see more detail but runtime grows linearly.
width = height = 1000

# Region of the complex plane to sample. The 'extent' passed to imshow must
# match these values so the axes display the correct complex coordinates.
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Generate coordinates along each axis (inclusive endpoints).
# x_vals maps to columns (real axis), y_vals maps to rows (imaginary axis).
x_vals = linspace(x_min, x_max, width)
y_vals = linspace(y_min, y_max, height)
print("Generated x and y values for the complex plane.")

# Preallocate a height x width storage. Each cell will hold either:
# - an integer 0..2 indicating which known root was reached, or
# - a complex number indicating no convergence within max_iter.
newton_fractal_plot = zeros((height, width))
print("Initialized newton_fractal_plot array.")


# Main computation: apply Newton iteration to each seed

# We iterate over rows (y) and columns (x). The mapping convention used here:
#   newton_fractal_plot[row][col] corresponds to seed = x_vals[col] + y_vals[row]*1j
for i in range(height):
    for j in range(width):
        seed = x_vals[j] + y_vals[i] * 1j
        newton_fractal_plot[i][j] = newton_fractal(seed)

print("Computed Newton fractal values for the grid.")


# Visualization
# Create a 16:9 figure for nicer presentation
plt.figure(figsize=(16, 9))

# imshow displays the 2D list. Important arguments:
# - extent maps pixel coordinates to (x_min,x_max,y_min,y_max)
# - interpolation='nearest' preserves the discrete basin boundaries
# - origin='lower' will place the first row at the bottom (mathematical y-up).
plt.imshow(
    newton_fractal_plot,
    extent=(x_min, x_max, y_min, y_max),
    interpolation='nearest',
    colorizer="tab10"
)

plt.title('Newton Fractal (f(x) = x^3 + 1)')
plt.xlabel('Re(c) - Real Axis')
plt.ylabel('Im(c) - Imaginary Axis')

# Optional: choose a colormap that makes the three basins distinct.
# Examples: 'tab10', 'viridis', 'plasma', 'twilight', 'hsv'
# plt.colormaps can be large; choose one appropriate for discrete integer labels.
# E.g. for discrete palette: cmap = plt.cm.get_cmap('tab10', 10)
# plt.imshow(..., cmap=plt.cm.get_cmap('tab10', 10), ...)

# Save the high-resolution image.
plt.savefig(
    r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q2b_newton_fractal.png",
    dpi=300
)
print("Saved Newton fractal image to disk.")