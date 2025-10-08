import math

#The function
def f(x):
    return 500-7.5*math.sqrt(x-25)

#error is maximum relative error 
error = 10**-4
#x is the initial guess, i.e. x0
#n is the number of iterations
x,n = 150,0
prevx = x

print ("x",n," = ",x)

x = f(x)
n = n+1
print ("x",n," = ",x)


while (abs(x-prevx) > error):
    prevx = x
    x = f(x)
    n = n+1

    print ("x",n,"=",x)

print ("Final approximation: ", x)
print ("Number of iterations: ", n)