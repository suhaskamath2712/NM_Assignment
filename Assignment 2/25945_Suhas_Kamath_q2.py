import numpy as np

#Manually implementing np.linalg.solve()
def solve_linear_equation(A, b):
    # Solve the linear equation Ax = b
    # Gaussian elimination
    M = A.astype(float).copy()
    bb = b.astype(float).copy()
    n = M.shape[0]

    # Forward elimination
    for k in range(n):
        # Partial pivoting
        piv = np.argmax(np.abs(M[k:, k])) + k
        if abs(M[piv, k]) < 1e-12:
            return None  #Error: Singular matrix, i.e. no unique solution
        if piv != k:
            M[[k, piv]] = M[[piv, k]]
            bb[[k, piv]] = bb[[piv, k]]
        # Eliminate below
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
            bb[i] -= factor * bb[k]

    # Back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (bb[i] - np.dot(M[i, i + 1:], x[i + 1:])) / M[i, i]
    return x

#table with initial data
table = [[0.0,150],[0.5,165],[1.2,172],[2.0,185]]

#data for which we want to interpolate
x = 1.0

#construct newton's divided difference table
for i in range(1,len(table)):
    for j in range(len(table)-i):
        table[j].append((table[j+1][i]-table[j][i])/(table[j+i][0]-table[j][0]))

print("Newton's Divided Difference Table:")
for row in table:
    print(row)

# compute polynomial coefficients in standard basis (a0 + a1 x + a2 x^2 + ...)
n = len(table)
coeffs = [0.0] * n
basis = [1.0]  # starts as polynomial 1

for i in range(n):
    ci = table[0][i+1]  # divided-difference coefficient for term i
    # add ci * basis to coeffs
    for k, b in enumerate(basis):
        coeffs[k] += ci * b
    # update basis *= (x - x_i) for next iteration
    if i < n - 1:
        xi = table[i][0]
        new_basis = [0.0] * (len(basis) + 1)
        for k, b in enumerate(basis):
            new_basis[k] += -xi * b
            new_basis[k + 1] += b
        basis = new_basis

poly_str = " + ".join(f"{coeff:.6g}x^{i}" for i, coeff in enumerate(coeffs))

print("Interpolating polynomial from Newton's divided difference method:", end=" ")
print("T(x) =", poly_str)

#function to evaluate polynomial at given x using the calculated coefficients
result = 0.0
for i, coeff in enumerate(coeffs):
    result += coeff * (x ** i)

print("Estimation by evaluating Newton's polynomial at x =", x, ":")

print("Estimation of f(x) at x =", x, ":", result)

print("------------------------------------------------------")

print("Verification by calculating interpolating polynomial as a system of equations:")
# Constructing the system of equations
A = np.zeros((n, n))
b = np.zeros(n)

for i in range(n):
    for j in range(n):
        A[i, j] = table[i][0] ** j
    b[i] = table[i][1]

# Solving the system
coeffs = solve_linear_equation(A, b)

# Formatting the polynomial string
poly_str = " + ".join(f"{coeff:.6g}x^{i}" for i, coeff in enumerate(coeffs) if coeff != 0)

print("Interpolating polynomial from solving linear equations:", end="\t")
print("T_p(x) =", poly_str)
