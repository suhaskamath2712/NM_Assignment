import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 1/(1+x**2)

def simpsons_rule(a, b, n):    
    h = (b - a) / n
    integral = f(a) + f(b)
    
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    
    for i in range(2, n-1, 2):
        integral += 2 * f(a + i * h)
    
    integral *= h / 3
    return integral

components = [4,8,16,32,64,128]
a, b = 0, 1

results = [simpsons_rule(a, b, n) for n in components]
error = [abs(math.pi/4 - result) for result in results]

#Print results
for n, result, err in zip(components, results, error):
    print(f"n={n}, Integral={result}, Error={err}")

#Plot error graph log-log scale
plt.plot(np.log(components), np.log(error), marker='o', label='Error Line')
plt.plot(np.log(components), -4*np.log(np.array(components))-5, label='Reference: y=-4x', linestyle='--')
plt.xlabel('log(Number of Subintervals (n))')
plt.ylabel('log(Error)')
plt.title("Error in Simpson's Rule")
plt.grid()
plt.legend()
plt.show()
