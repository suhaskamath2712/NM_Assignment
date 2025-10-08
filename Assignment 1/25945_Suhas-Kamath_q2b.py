import math
import matplotlib.pyplot as plt
import numpy as np

#The function
def f(x):
    return math.atan(2*x+3)+1

#The first derivative of the function
def fprime(x):
    denom=(2*x+3)**2+1
    return 2/denom

#The second derivative of the function
def fprimeprime(x):
    num = -8*(2*x+3)
    denom = ((2*x+3)**2+1)**2
    return num/denom

def calc_rel_error(x, prevx):
    return abs((x-prevx)/x)


#error is maximum relative error 
error = 10**-7
newtons_method_errors = []
newtons_modified_method_errors = []
#x is the initial guess, i.e. x0
#n is the number of iterations
x,n = -1,0
prevx = x

print("Using Newton's method: ")
print ("x",n," = ",x)

x = prevx - f(prevx)/fprime(prevx)
n = n+1


while (calc_rel_error(x,prevx) > error):
    newtons_method_errors.append(calc_rel_error(x,prevx))
    prevx = x
    x = prevx - f(prevx)/fprime(prevx)
    n = n+1

    print ("x",n,"=",x)

print ("Final approximation: ", x)
print ("Number of iterations: ", n)

print(newtons_method_errors)

#Reset values of x & n
x,n = -1,0
prevx = x

print("Using Newton's modified method: ")
print ("x",n," = ",x)


#Calculate numerator and denominator separately
num = f(prevx)*fprime(prevx)
denom = (fprime(prevx))**2-f(prevx)*fprimeprime(prevx)
x = prevx - num/denom
n = n+1

while (calc_rel_error(x,prevx) > error):
    newtons_modified_method_errors.append(calc_rel_error(x,prevx))
    prevx = x
    num = f(prevx)*fprime(prevx)
    denom = (fprime(prevx))**2-f(prevx)*fprimeprime(prevx)
    x = prevx - num/denom
    n = n+1

    print ("x",n,"=",x)

print ("Final approximation: ", x)
print ("Number of iterations: ", n)

#plotting convergence of Newton's method
plt.plot(np.log(newtons_method_errors[:-1]),np.log(newtons_method_errors[1:]), label="Actual Line")
plt.plot(np.log(newtons_method_errors[:-1]),np.log(newtons_method_errors[:-1])*2, linestyle="--", label="Reference Line: y = 2x")
plt.xlabel("log (nth error)")
plt.ylabel("log ((n+1)th error)")
plt.legend()
plt.grid()
plt.title("Question 2(b) \n Convergence of Newton's method")
plt.show()

#plotting convergence of Newton's modified method
plt.plot(np.log(newtons_modified_method_errors[:-1]),np.log(newtons_modified_method_errors[1:]), label="Actual Line")
plt.plot(np.log(newtons_modified_method_errors[:-1]),np.log(newtons_modified_method_errors[:-1])*2, linestyle="--", label="Reference Line: y = 2x")
plt.xlabel("log (nth error)")
plt.ylabel("log ((n+1)th error)")
plt.legend()
plt.grid()
plt.title("Question 2(b) \n Convergence of Newton's modified method")
plt.show()

#plotting convergence of Newton's vs. modified Newton's method
plt.plot(np.log(newtons_method_errors),label="Errors using Newton's method")
plt.plot(np.log(newtons_modified_method_errors),label="Errors using Newton's modified method")
plt.xlabel("Iterations")
plt.ylabel("log (Error)")
plt.legend()
plt.grid()
plt.title("Question 2(b) \n Comparison of the methods")
plt.show()