import math
import matplotlib.pyplot as plt

def f(x):
    return math.exp(-x**2)

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n):
        integral += 2*f(a + i * h)
    
    integral *= h/2
    return integral

print("Part (b):")
a_vals = [0.5,1,2,5]
n_vals = [10,50,500,3000]
results = []

for a in a_vals:
    for n in n_vals:
        results.append((a, n, trapezoidal_rule(-a, a, n)))

for res in results:
    print(f"a: {res[0]:.4f}, n: {res[1]}, Integral: {res[2]:.4f}")

print("====================================================================")
print("Part (c):")
exact = math.pi**0.5

#Print error table
for result in results:
    error = abs(exact - result[2])
    print(f"a: {result[0]:.4f}, n: {result[1]}, Integral: {result[2]:.4f}, Error: {error:.6f}")

print("====================================================================")
print("Part (e):")
errors = [abs(trapezoidal_rule(-a/10, a/10, 3000) - exact) for a in range(0,200)]
plt.plot([a/10 for a in range(0,200)], errors)
plt.yscale('log')
plt.xlabel('a')
plt.ylabel('Absolute Error')
plt.title('Error vs a for n=3000')
plt.grid()
plt.show()