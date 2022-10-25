#!/usr/bin/env python
# coding: utf-8

# In[64]:


# This python module contains the convertion of MATLAB code, developed for
# the course ENGG 818: Advanced Numerical Methods of University of Regina.
#
# All rights reserve to Professor Wei Peng and Surya Pusapati.
#
#
# NOTES:
# Add arguments into equation
# Reduce the float decimals in the function output print. To make the output look good.
# Incomplete: Gauss_Newton, Multiple_Linear, Newint2.


# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def bisect(func, xl, xu, es = 1e-4, maxit = 50, *arg):
    '''
    bisect:
    root location zeros
    uses bisection method to find the root of func
    input:
    func = name of function
    xl, xu = lower and upper guesses
    es = desired relative error (default = 0.0001%)
    maxit = maximum allowable iterations (default = 50)
    p1,p2,... = additional parameters used by func
    output:
    root = real root
    fx = function value at root
    ea = approximate relative error (%)
    iter = number of iterations
    '''
    test = func(xl)*func(xu)
    if test > 0:
        raise Exception("no sign change")
    iter_ = 0
    xr = xl
    ea = 100
    while(True):
        xrold = xr
        xr = (xl + xu)/2
        iter_ += 1
        if xr != 0:
            ea = abs((xr - xrold)/xr)*100
        test = func(xl)*func(xr)
        if test < 0:
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0
        if ea <= es and iter_ >= maxit:
            break
    root = xr
    fx = func(xr)
    return root, fx, ea, iter_


# In[5]:


def bisect2(func):
    xl = float(input('enter lower bound xl = '))
    xu = float(input('enter upper bound xu = '))
    es = float(input('allowable tolerance es = '))
    maxit = int(input('maximum number of iteration maxit = '))
    x, y, a, b, ya, yb = list(), list(), list(), list(), list(), list()
    if xl > xu:
        raise Exception('\'xl\' should be less than \'xu\'')
    a.append(xl)
    b.append(xu)
    ya.append(func(a[0]))
    yb.append(func(b[0]))
    if (ya[0]*yb[0]) > 0.0:
        raise Exception('Function has same sign at end points')
    for i in range(maxit):
        x.append((a[i]+b[i])/2)
        y.append(func(x[i]))
        if (x[i]-a[i]) < es:
            print('Bisection method has converged')
            break
        if y[i] == 0.0:
            print('exact zero found')
            break
        elif (y[i]*ya[i]) < 0:
            a.append(a[i])
            ya.append(ya[i])
            b.append(x[i])
            yb.append(y[i])
        else:
            a.append(x[i])
            ya.append(y[i])
            b.append(b[i])
            yb.append(yb[i])
    
    if i >= maxit:
        print('zero not found to desired tolerance')

    print('\tstep\txl\txu\txr\tf(xr)')
    for i in range(len(x)):
        print("\t{}\t{}\t{}\t{}\t{}".format(i+1, a[i], b[i], x[i], y[i]))
    return x, y


# In[8]:


def false_position(func):
    xl = float(input('enter lower bound xl = '))
    xu = float(input('enter upper bound xu = '))
    es = float(input('allowable tolerance es = '))
    maxit = int(input('maximum number of iteration maxit = '))
    x, y, a, b, ya, yb = list(), list(), list(), list(), list(), list()
    if xl > xu:
        raise Exception('\'xl\' should be less than \'xu\'')
    a.append(xl)
    b.append(xu)
    ya.append(func(a[0]))
    yb.append(func(b[0]))
    if (ya[0]*yb[0]) > 0.0:
        raise Exception('Function has same sign at end points')

    for i in range(maxit):
        x.append(b[i]-yb[i]*(b[i]-a[i])/(yb[i]-ya[i]))
        y.append(func(x[i]))
        if y[i] == 0.0:
            print('exact zero found')
            break
        elif (y[i]*ya[i]) < 0:
            a.append(a[i])
            ya.append(ya[i])
            b.append(x[i])
            yb.append(y[i])
        else:
            a.append(x[i])
            ya.append(y[i])
            b.append(b[i])
            yb.append(yb[i])
        if (i > 1 and (abs(x[i]-x[i-1]) < es)):
            print('False position method has converged')
            break
    if i >= maxit:
        print('zero not found to desired tolerance')

    print('\tstep\txl\txu\txr\tf(xr)')
    for i in range(len(x)):
        print("\t{}\t{}\t{}\t{}\t{}".format(i+1, a[i], b[i], x[i], y[i]))
    return x, y


# In[11]:


def newtraph(func, dfunc, xr, es = 1e-4, maxit = 50, *arg):
    '''
    newtraph: Newton-Raphson root location zeroes
    [root,ea,iter]=newtraph(func,dfunc,xr,es,maxit,p1,p2,...):
    uses Newton-Raphson method to find the root of func
    input:
    func = name of function
    dfunc = name of derivative of function
    xr = initial guess
    es = desired relative error (default = 0.0001%)
    maxit = maximum allowable iterations (default = 50)
    p1,p2,... = additional parameters used by function
    output:
    root = real root
    ea = approximate relative error (%)
    i = number of iterations
    '''
    i = 0
    while(True):
        xrold = xr
        xr = xr - func(xr)/dfunc(xr)
        i += 1
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100
        if ea <= es and i >= maxit:
            break
    root = xr
    return root, ea, i


# In[13]:


def multiple(func, dfunc):
    m = float(input('enter multiplicity of the root = '))                        # check
    xguess = float(input('enter initial guess: xguess = '))
    es = float(input('allowable tolerance es = '))
    maxit = int(input('maximum number of iterations: maxit = '))
    x, f, dfdx = list(), list(), list()
    x.append(xguess)
    f.append(func(x[0]))
    dfdx.append(dfunc(x[0]))
    for i in range(maxit):
        x.append(x[i]-m*f[i]/dfdx[i])
        f.append(func(x[i]))
        dfdx.append(dfunc(x[i]))
        if abs(x[i+1]-x[i])<es:
            print('Newton method has converged')
            break

    if i >= maxit:
        print('zero not found to desired tolerance')
    
    print('\tstep\tx\tf\tdf/dx')
    for i in range(len(x)):
        print("\t{}\t{}\t{}\t{}".format(i+1, x[i], f[i], dfdx[i]))
    return x, f


# In[17]:


def multiple2(func, dfunc, ddfunc):
    xguess = float(input('enter initial guess: xguess = '))
    es = float(input('allowable tolerance es ='))
    maxit = int(input('maximum number of iterations: maxit = '))
    x, f, dfdx, d2fdx2 = list(), list(), list(), list()
    x.append(xguess)
    f.append(func(x[0]))
    dfdx.append(dfunc(x[0]))
    d2fdx2.append(ddfunc(x[0]))
    for i in range(maxit):
        x.append(x[i] - f[i]*dfdx[i]/(dfdx[i]**2-f[i]*d2fdx2[0]))
        f.append(func(x[i]))
        dfdx.append(dfunc(x[i]))
        d2fdx2.append(ddfunc(x[i]))
        if abs(x[i+1]-x[i])<es:
            print('Newton method has converged')
            break

    if i >= maxit:
        print('zero not found to desired tolerance')

    print('\tstep\tx\tf\tdf/dx\td2f/dx2')
    for i in range(len(x)):
        print("\t{}\t{}\t{}\t{}\t{}".format(i+1, x[i], f[i], dfdx[i], d2fdx2[i]))
    return x, f


# In[65]:


def Cubic_LS(x, y):
    '''
    Cubic Least Squares regression function
    input x and y as row or column vectors
    (they are converted to column form if necessary)
    '''
    n = len(x)
    sx = sum(x)
    sx2 = sum(x**2)
    sx3 = sum(x**3)
    sx4 = sum(x**4)
    sx5 = sum(x**5)
    sx6 = sum(x**6)
    sy = sum(y)
    syx = sum(x*y)
    syx2 = sum(y*x**2)
    syx3 = sum(y*x**3)
    A = [[n, sx, sx2, sx3],
         [sx, sx2, sx3, sx4],
         [sx2, sx3, sx4, sx5],
         [sx3, sx4, sx5, sx6]]
    r = [sy, syx, syx2, syx3]
    z = np.linalg.solve(A, r)
    a0 = z[0]
    a1 = z[1]
    a2 = z[2]
    a3 = z[3]
    p = a0 + a1*x + a2*x**2 + a3*x**3
    print('\tstep\tx\ty\tp(x) = a0 +a1*x + a2*x.^2 + a3*x.^3\ty-p(x)')
    for i in range(len(x)):
        print("\t{}\t{}\t{}\t{}\t{}\n".format(i+1, x[i], y[i], p[i], (y[i]-p[i]))) 
    err = sum((y-p)**2)

    St = sum((y-sy/n)**2)
    Sr = err
    Syx = np.sqrt(Sr/(n-3)) # standard error of the estimate
    r = np.sqrt((St-Sr)/St) # correlation coefficient

    #plot the regression curve
    x1=min(x)
    x2=max(x)
    xx = np.linspace(x1, x2, 50)
    yy = a0+a1*xx+a2*xx**2+a3*xx**3
    plt.plot(x,y,'r*',xx,yy,'b-')
    plt.show()
    return z, Syx, r


# In[71]:


def Gauss_Newton(x,y):
    '''
    Nonlinear Regression of f(x)=exp(-a0*x)cos(a1*x) using Gauss-Newton method
    '''
    a = [int(input('Enter the initial guesses a0 = ')), int(input('Enter the initial guesses a1 = '))]
    tol = float(input('Enter the tolerance to1 = '))
    itmax = int(input('Enter the maximum iteration number itmax = '))
    
    n = len(x)
    f = np.empty(shape=n)
    d = np.empty(shape=n)
    da = np.empty(shape=n)
    print('\titer\ta0\ta1\tda0\tda1')
    for iter_ in range(itmax):
        a0 = a[0]
        a1 = a[1]
        z = np.empty(shape=(n,2))
        for i in range(n):
            f[i] = np.exp(-a0*x[i])*np.cos(a1*x[i])
            d[i] = y[i] - f[i]
            z[i,0] = -x[i]*np.exp(-a0*x[i])*np.cos(a1*x[i])
            z[i,1] = -x[i]*np.exp(-a0*x[i])*np.sin(a1*x[i])
        da = np.linalg.solve(z.conj().T@z, (z.conj().T*d.conj().T))
        a += da.conj().T
        print("\t{}\t{}\t{}\n".format(iter_+1, a, da.conj().T)) 
        if (abs(da[0]) < tol & abs(da[1]) < tol):
            print('Gauss-Newton method has converged')
    
    x1=min(x)
    x2=max(x)
    xx = np.linspace(x1, x2, 50)
    yy = np.exp(-a0*xx)*np.cos(a1*xx)
    plt.plot(xx,yy,x,y,'ro')
    #set(H,'LineWidth',2,'MarkerSize',7);
    return a


# In[3]:


def Lagrange_coef(x,y):
    # calculate coefficients of Lagrange funcitons
    n = len(x)
    d = np.empty(shape=n)
    c = np.empty(shape=n)
    for k in range(n):
        d[k]=1
        for i in range(n):
            if i != k:
                d[k]=d[k]*(x[k]-x[i])
            c[k]=y[k]/d[k]
    return c


# In[24]:


def Lagrange_Eval(t,x,c):
    # Evaluate Lagrange interplation polynomial at x = t
    m = len(x)
    n = len(t)
    p = np.empty(shape=n)
    N = np.empty(shape=m)
    for i in range(n):
        p[i] = 0
        for j in range(m):
            N[j] = 1
            for k in range(m):
                if j != k:
                    N[j] = N[j]*(t[i]-x[k])
            p[i] = p[i]+N[j]*c[j]
    return p


# In[33]:


def Linear_LS(x,y):
    # linear regression function
    # input x and y as row or column vectors
    # (they are converted to column form if necessary)
    m=len(x)
    sx=sum(x)
    sy=sum(y)
    sxx=sum(x*x)
    sxy= sum(x*y)
    a0=(sxx*sy-sxy*sx)/(m*sxx-sx**2)
    a1=(m*sxy-sx*sy)/(m*sxx-sx**2)
    print('\tx\ty\t(a0+a1*x)\t(y-a0-a1*x)')
    for i in range(m):
        print("\t{}\t{}\t{}\t{}\t{}\n".format(i+1, x[i], y[i], a0+a1*x[i], y[i]-a0-a1*x[i]))
    err=sum((y-a0-a1*x)**2)
    St=sum((y-sy/m)**2)
    Sr=err
    Syx=np.sqrt(Sr/(m-2))  # standard error of the estimate
    r=np.sqrt((St-Sr)/St) # correlation coefficient
    # plot the regression curve
    x1=min(x)
    x2=max(x)
    xx=np.linspace(x1, x2, 50)
    yy=a0+a1*xx
    plt.plot(x,y, 'r*', xx,yy, 'g')
    plt.show()
    return [a1, a0], r


# In[35]:


def linregr(x,y):
    # linregr: linear regression curve fitting
    #   [a, r2] = linregr(x,y): Least squares fit of straight
    #           line to data by solving the normal equations
    # input:
    #   x = independent variable
    #   y = dependent variable
    # output:
    #   a = vector of slope, a(1), and intercept, a(2)
    #   r2 = coefficient of determination
    n = len(x)
    #if len(y)!=n:
        # error('x and y must be same length')                               #check
    sx = sum(x)
    sy = sum(y)
    sx2 = sum(x*x)
    sxy = sum(x*y)
    sy2 = sum(y*y)
    a0 = (n*sxy-sx*sy)/(n*sx2-sx**2)
    a1 = sy/n-a0*sx/n
    r2 = ((n*sxy-sx*sy)/np.sqrt(n*sx2-sx**2)/np.sqrt(n*sy2-sy**2))**2
    # create plot of data and best fit line
    xp = np.linspace(min(x), max(x), 50)
    yp = a0*xp+a1
    plt.plot(x,y,'o',xp,yp)
    plt.grid()
    return [a0, a1], r2


# In[69]:


def Multiple_Linear(x1,x2,y):
    # Multiple variable Least Squares regression function
    # input x and y as row or column vectors

    n = len(x1)
    sx1 = sum(x1)
    sx2 = sum(x2)
    sx1x2 = sum(x1*x2)
    sx1x1 = sum(x1**2)
    sx2x2 = sum(x2**2)
    sy = sum(y)
    sx1y = sum(x1*y)
    sx2y = sum(x2*y)
    A = np.array([[n, sx1, sx2],
                 [sx1, sx1x1, sx1x2],
                 [sx2, sx1x2, sx2x2]])
    r = [sy, sx1y, sx2y]
    z = np.linalg.solve(A, r)
    a0 = z[0]
    a1 = z[1]
    a2 = z[2]
    
    print('\tx1\tx2\ty\t(a0+a1*x1+a2*x2)\t(y-a0-a1*x1-a2*x2)')
    for i in range(n):
        print("\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i+1, x1[i], x2[i], y[i], (a0+a1*x1[i]+a2*x2[i]), (y[i]-a0-a1*x1[i]-a2*x2[i])))
    
    err = sum((y-a0-a1*x1-a2*x2)**2)
    St = sum((y-sy/n)**2)
    Sr = err
    Syx = np.sqrt(Sr/(n-3)) # standard error of the estimate
    r = np.sqrt((St-Sr)/St) # correlation coefficient

    # plot the experimental data and regression plane
    #H = plot3(x1,x2,y,'ro')
    #grid on
    #set(H,'LineWidth',6)
    #H1=xlabel('cure time (days)')
    #set(H1,'FontSize',12)
    #H2=ylabel('Water Content')
    #set(H2,'FontSize',12)
    #H3=zlabel('Strength (psi)')
    #set(H3,'FontSize',12)
    #hold on
    #x1a=min(x1)
    #x1b=max(x1)
    #x1s=x1a:(x1b-x1a)/50:x1b
    #x2a=min(x2)
    #x2b=max(x2)
    #x2s=x2a:(x2b-x2a)/50:x2b
    #[xx1,xx2]=meshgrid(x1s,x2s)
    #yy=a0+a1*xx1+a2*xx2
    #surf(xx1,xx2,yy)
    #hold off
    return z, r



# In[68]:


def Newtint2(x,y,xx):
    '''
    # Newtint: Newton interpolating polynomial
    # yint = Newtint(x,y,xx): Uses an (n - 1)-order Newton
    #   interpolating polynomial based on n data points (x, y)
    #   to determine a value of the dependent variable (yint)
    #   at a given value of the independent variable, xx.
    # input:
    #   x = independent variable
    #   y = dependent variable
    #   xx = value of independent variable at which
    #        interpolation is calculated
    # output:
    #   yint = interpolated value of dependent variable

    # compute the finite divided differences in the form of a
    # difference table
    n = len(x)
    #if len(y)!=n:
        #error('x and y must be same length'); end                                     #check
    b = np.zeros(n,n)
    # assign dependent variables to the first column of b.
    b(:,1) = y(:); # the (:) ensures that y is a column vector.
    for j in range(1,n):
        for i = 1:n-j+1
        b(i,j) = (b(i+1,j-1)-b(i,j-1))/(x(i+j-1)-x(i))
        end
    end
    # use the finite divided differences to interpolate
    for k =1:len(xx)
    xt = 1
    yint(k) = b(1,1)
    for j = 1:n-1
        xt = xt*(xx(k)-x(j))
        yint(k) = yint(k)+b(1,j+1)*xt
    end
    end
    '''
    return b, yint


# In[51]:


def quadratic(x, f):
    # exact solution f(x)=x^3-5x^2+3x+4
    # x=[-1 0 2 5 6]; f=[-5 4 -2 19 58];
    h1=x[1]-x[0]
    h2=x[2]-x[1]
    h3=x[3]-x[2]
    h4=x[4]-x[3]
    f1=f[0]
    f2=f[1]
    f3=f[2]
    f4=f[3]
    f5=f[4]
    A=np.array([[1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
                [0,    h1, h1**2,  0,    0,    0,    0,    0,    0,    0,    0,    0],
                [0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],
                [0,    0,    0,    0,   h2,  h2**2,  0,    0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,   h3,  h3**2,  0,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    h4, h4**2],
                [0,    1,   2*h1,  0,    -1,   0,    0,    0,    0,    0,    0,    0],
                [0,    0,    0,    0,    1,  2*h2,   0,   -1,    0,    0,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,    1,  2*h3,   0,   -1,    0],
                [0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0]])
    b=[f1, f2-f1, f2, f3-f2, f3, f4-f3, f4, f5-f4, 0, 0, 0, 0]
    return A, b


# In[63]:


def Quadratic_LS(x,y):
    # Quandratic Least Squares regression function
    # input x and y as row or column vectors
    # (they are converted to column form if necessary)

    n = len(x)
    sx=sum(x)
    sx2=sum(x**2)
    sx3=sum(x**3)
    sx4=sum(x**4)
    sy=sum(y)
    sxy=sum(x*y)
    sx2y=sum(x*x*y)
    A=np.array([[n, sx, sx2],
                [sx, sx2, sx3],
                [sx2, sx3, sx4]])
    r=[sy, sxy, sx2y]
    z=np.linalg.solve(A, r)
    a0=z[0]
    a1=z[1]
    a2=z[2]
    
    print('\tx\ty\t(a0+a1*x+a2*x**2)\t(y-a0-a1*x-a2*x**2)')
    for i in range(n):
        print("\t{}\t{}\t{}\t{}\t{}\n".format(i+1, x[i], y[i], a0+a1*x[i]+a2*x[i]**2, y[i]-a0-a1*x[i]-a2*x[i]**2))

    err = sum((y-a0-a1*x-a2*x**2)**2)
    St = sum((y-sy/n)**2)
    Sr = err
    Syx = np.sqrt(Sr/(n-3)) # standard error of the estimate
    r = np.sqrt((St-Sr)/St) # correlation coefficient

    # plot the regression curve
    x1=min(x)
    x2=max(x)
    xx=np.linspace(x1, x2, 50)
    yy=a0+a1*xx+a2*xx**2
    plt.plot(x,y,'r*',xx,yy,'m')
    plt.show()
    return z, Syx, r

