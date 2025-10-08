# The function for which we need to find the root
def f(h):
    return (h**2)*(1.5-h)-12/41

#The signum function, to return the sign of the number
def signum(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

#a & b represent the intial range
#n represents the number of iterations
#error represents the absolute error (epsilon) 
a,b,n,error = 0,1,0,10**-4

#Check if the solution exists
#Although we have checked this manually, there is no harm in checking again
if signum(f(a))*signum(f(b)) < 0:
    print("Solution is there")
    prevx = x = (a+b)/2
    print ("x",n," = ",x)

    if signum(f(x))*signum(f(b)) < 0:
        a = x
    elif signum(f(x))*signum(f(a)) < 0:
        b = x
    
    x = (a+b)/2
    n = n+1

    print ("x",n," = ",x)

    #Keep looping until the error is not within limit (epsilon)
    while (abs(x-prevx) > error):
        prevx = x
        if signum(f(x))*signum(f(b)) < 0:
            a = x
        elif signum(f(x))*signum(f(a)) < 0:
            b = x
        x = (a+b)/2
        n = n+1

        print ("x",n,"=",x)
    
    #Print final approximation & number of iterations required
    print ("Final approximation: ", x)
    print ("Number of iterations: ", n)


else:
    print("No solution")

