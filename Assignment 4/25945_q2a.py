"""
Mandelbrot set visualization using pure Python lists and matplotlib.

Overview:
- Samples a rectangular region of the complex plane on a width x height grid.
- For each point c in the plane, iterates z_{n+1} = z_n^2 + c starting at z_0 = 0.
- Records the iteration count at which |z_n| exceeds the bailout radius (2), up to max_iter.
- Renders the escape-time image with matplotlib and saves it to disk.

- Computational complexity is O(width * height * max_iter). For 1000x1000 and max_iter=100,
  this is ~100 million complex ops; expect noticeable runtime in CPython.
"""

import matplotlib.pyplot as plt

# Mandelbrot iteration function
def mandelbrot(c, max_iter=100):
    """
    Compute the escape iteration count for a complex parameter c.

    Algorithm:
    - Start with z = 0.
    - Iterate z <- z*z + c while |z| <= 2 and iteration count < max_iter.
    - Return the number of iterations performed (escape-time). If the point
      does not escape within max_iter, the return value equals max_iter.

    Args:
        c (complex): Point in the complex plane to test.
        max_iter (int): Maximum number of iterations to attempt.

    Returns:
        int: Iteration count at escape (<= max_iter).
    """
    z = 0 + 0j  # initial condition z0 = 0
    n = 0       # iteration counter
    
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n

# Custom implementation of linspace (inclusive of both start and stop)
def linspace(start, stop, num):
    """
    Generate 'num' evenly-spaced values from start to stop inclusive.

    Caveats:
    - Floating-point rounding may lead to small endpoint inaccuracies.
    - For large 'num', prefer numpy.linspace for performance and robustness.

    Args:
        start (float): Start value (included).
        stop (float): Stop value (included).
        num (int): Number of samples to generate (>= 1).

    Returns:
        list[float]: List of sampled values.
    """
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]

# Custom 1D/2D zeros helper returning nested Python lists
def zeros(shape):
    """
    Create a 1D or 2D list filled with zeros.

    Args:
        shape (tuple[int] | list[int]): (n,) for 1D or (rows, cols) for 2D.

    Returns:
        list: A list of zeros (1D) or list of lists (2D).

    Raises:
        ValueError: If shape has rank other than 1 or 2.

    Note:
        For numerical work, numpy.zeros is more efficient and convenient.
    """
    if len(shape) == 1:
        return [0] * shape[0]
    elif len(shape) == 2:
        return [[0] * shape[1] for _ in range(shape[0])]
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")

# Parameters of the visualization grid
# 'width' and 'height' control both sampling density and output image resolution.
width = height = 1000

# Rectangular region of the complex plane to visualize:
# x-axis corresponds to Re(c), y-axis to Im(c).
x_min, x_max = -2.5, 1.0
y_min, y_max = -1.5, 1.5

# Generate the sample coordinates along each axis.
# linspace is inclusive, so you get exactly 'width' and 'height' points.
x_vals = linspace(x_min, x_max, width)
y_vals = linspace(y_min, y_max, height)
print("Generated x and y values for the complex plane.")

# Preallocate the 2D escape-time array (height rows by width columns).
# Each entry will store the iteration count (0..max_iter).
mandelbrot_plot = zeros((height, width))
print("Initialized mandelbrot_plot array.")

# Compute escape times for each pixel (i, j).
# Mapping: row i -> y coordinate y_vals[i], column j -> x coordinate x_vals[j].
# Total work \approx width * height * average_iterations.
for i in range(height):
    for j in range(width):
        mandelbrot_plot[i][j] = mandelbrot(x_vals[j] + y_vals[i] * 1j)

print("Computed Mandelbrot set values for the grid.")

# Visualization using matplotlib
plt.figure(figsize=(16, 9))
plt.imshow(
    mandelbrot_plot,
    extent=(x_min, x_max, y_min, y_max))
plt.title('Mandelbrot Set')
plt.xlabel('Re(c) - Real Axis')
plt.ylabel('Im(c) - Imaginary Axis')

# Save the figure at high resolution.
plt.savefig(
    r"H:\My Drive\Numerical Methods\Assignments\Assignment 4\q2a_mandelbrot.png"
)
print("Saved Mandelbrot set image to disk.")
