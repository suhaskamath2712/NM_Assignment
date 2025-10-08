import matplotlib.pyplot as plt


def hermite_newton(x, fx, fpx):
	"""Compute Hermite interpolating polynomial in Newton divided-difference form.

	Returns:
	  z: list of repeated nodes (length 2n)
	  coeffs: list of Newton-form coefficients (length 2n), so polynomial is
			  P(x) = coeffs[0] + coeffs[1]*(x-z[0]) + coeffs[2]*(x-z[0])*(x-z[1]) + ...
	  Q: full divided difference table (for debugging/printing)
	"""
	n = len(x)
	m = 2 * n
	z = [0] * m
	Q = [[0.0 for _ in range(m)] for _ in range(m)]

	for i in range(n):
		z[2 * i] = x[i]
		z[2 * i + 1] = x[i]
		Q[2 * i][0] = fx[i]
		Q[2 * i + 1][0] = fx[i]
		Q[2 * i + 1][1] = fpx[i]
		if i != 0:
			Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1])

	for j in range(2, m):
		for i in range(j, m):
			Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])

	coeffs = [Q[i][i] for i in range(m)]
	return z, coeffs, Q


def poly_mul(p, q):
	"""Multiply two polynomials p and q given as coefficient lists."""
	r = [0.0] * (len(p) + len(q) - 1)
	for i, pv in enumerate(p):
		for j, qv in enumerate(q):
			r[i + j] += pv * qv
	return r


def poly_add(p, q):
	L = max(len(p), len(q))
	r = [0.0] * L
	for i in range(L):
		if i < len(p):
			r[i] += p[i]
		if i < len(q):
			r[i] += q[i]
	return r


def newton_to_power(z, coeffs):
	"""Convert Newton basis polynomial (with nodes z and Newton coeffs) to power basis.

	Returns list of power-basis coefficients [a0, a1, a2, ...] so P(x)=a0 + a1 x + a2 x^2 + ...
	"""
	m = len(coeffs)
	power = [0.0] * m
	for k in range(m):
		# build product (x - z[0])...(x - z[k-1]) as polynomial
		term = [1.0]
		for j in range(k):
			term = poly_mul(term, [-z[j], 1.0])
		# scale by coeffs[k]
		term = [c * coeffs[k] for c in term]
		power = poly_add(power, term)
	# zero-out tiny values
	for i in range(len(power)):
		if abs(power[i]) < 1e-14:
			power[i] = 0.0
	return power


def format_polynomial(power_coeffs):
	"""Format polynomial in standard human-readable form: a + b x + c x^2 + ..."""
	parts = []
	for i, a in enumerate(power_coeffs):
		if a == 0:
			continue
		# Use compact formatting
		s = f"{a:.6g}"
		if i == 0:
			parts.append(s)
		elif i == 1:
			parts.append(f"{s} x")
		else:
			parts.append(f"{s} x^{i}")
	if not parts:
		return "0"
	# join and fix signs like '+ -'
	expr = ' + '.join(parts)
	expr = expr.replace('+ -', '- ')
	return expr


def evaluate_power(poly, xval):
	s = 0.0
	for i, a in enumerate(poly):
		s += a * (xval ** i)
	return s

if __name__ == '__main__':
	x = [0, 3, 5, 8, 13]
	fx = [0, 225, 383, 623, 993]
	fpx = [75, 77, 80, 74, 72]

	z, newton_coeffs, Q = hermite_newton(x, fx, fpx)

	print('Divided difference table (row i shows Q[i][j] for j=0..):')
	for i, row in enumerate(Q):
		entries = [f"{val:10.4f}" for val in row[:i+1]]
		print(f"i={i:2d}: ", '  '.join(entries))


	power = newton_to_power(z, newton_coeffs)


	print('\nHermite polynomial (standard form):')
	print(format_polynomial(power))

	# evaluate at x=10
	x_eval = 10
	val = evaluate_power(power, x_eval)
	print(f"\nP({x_eval}) = {val:.12g}")

	# use the divided difference to find the approximate speed at t = 10.
	h = 1e-4    #step size
	dx = (evaluate_power(power, x_eval + h) - evaluate_power(power, x_eval)) / h
	print(f"Approximate derivative P'({x_eval}) = {dx:.12g}")

	# small plot of polynomial and data points
	xs_plot = [i for i in range(0, 14)]
	ys_plot = [evaluate_power(power, xv) for xv in xs_plot]

	plt.plot(xs_plot, ys_plot, label='Hermite Polynomial')
	plt.scatter(x, fx, color='red', label='Data points')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('P(x)')
	plt.title('Hermite Interpolating Polynomial')
	plt.grid(True)
	plt.show()

	#=================================Part (b) starts here=============================
	
	# first derivative of the hermite polynomial
	power_derivative = [i * a for i, a in enumerate(power)][1:]
	
    #print hermite polynomial first derivative
	print('\nHermite polynomial first derivative (standard form):')
	print(format_polynomial(power_derivative))

	#print hermite polynomial second derivative
	power_second_derivative = [i * a for i, a in enumerate(power_derivative)][1:]
	print('\nHermite polynomial second derivative (standard form):')
	print(format_polynomial(power_second_derivative))

	#print hermite polynomial third derivative
	power_third_derivative = [i * a for i, a in enumerate(power_second_derivative)][1:]
	print('\nHermite polynomial third derivative (standard form):')
	print(format_polynomial(power_third_derivative))

	epsilon = 1e-6
	n = 1
	x0 = 14
	x1 = x0 - evaluate_power(power_second_derivative, x0) / evaluate_power(power_third_derivative, x0)
	#find roots of second derivative of hermite polynomial without using numpy
	#use newton's method, and starting value of x = 13
	while abs(x1 - x0) > epsilon:
		x0 = x1
		x1 = x0 - evaluate_power(power_second_derivative, x0) / evaluate_power(power_third_derivative, x0)
		n += 1
	
	print(f"\nRoot of second derivative: {x1:.12g} was calculated in {n} iterations")
	print(f"Value of hermite polynomial at root: {evaluate_power(power, x1):.12g}")
	print(f"Value of first derivative at root: {evaluate_power(power_derivative, x1):.12g}")
	print(f"Value of second derivative at root: {evaluate_power(power_second_derivative, x1):.12g}")
	print(f"Value of third derivative at root: {evaluate_power(power_third_derivative, x1):.12g}")
