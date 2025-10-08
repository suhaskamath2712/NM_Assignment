def f(x):
    return 4*(x**3)-300*x+600

def signum(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

a,b,n = 1,5,0
error = 10**-6

#check soln
if signum(f(a))*signum(f(b)) < 0:
    print("Solution is there")
    prevprevx = prevx = x = b - f(b)*((b-a)/(f(b)-f(a)))
    print ("x",n," = ",x)

    if signum(f(x))*signum(f(b)) < 0:
        a = x
    elif signum(f(x))*signum(f(a)) < 0:
        b = x
    
    prevx = x = b - f(b)*((b-a)/(f(b)-f(a)))
    n = n+1
    print ("x",n," = ",x)

    if signum(f(x))*signum(f(b)) < 0:
        a = x
    elif signum(f(x))*signum(f(a)) < 0:
        b = x

    x = b - f(b)*((b-a)/(f(b)-f(a)))
    n = n+1
    print ("x",n," = ",x)

    while (abs(x-prevx) > error):
        prevx = x
        if signum(f(x))*signum(f(b)) < 0:
            a = x
        elif signum(f(x))*signum(f(a)) < 0:
            b = x
        x = b - f(b)*((b-a)/(f(b)-f(a)))
        n = n+1

        print ("x",n,"=",x)
    
    print ("Final approximation: ", x)
    print ("Number of iterations: ", n)


else:
    print("No solution")