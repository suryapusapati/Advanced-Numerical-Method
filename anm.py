#!/usr/bin/env python
# coding: utf-8

# In[1]:

'''
----------------------------------------
    Advanced Numerical Methods (ANM)
----------------------------------------

This python module contains the convertion of MATLAB code of the course ENGG 818: Advanced Numerical Methods.

All rights reserve to Professor Wei Peng and Surya Pusapati.

'''


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint


# In[3]:


# code1


# In[4]:


def bisect(func, xl, xu, es=0.0001, maxit=50, *args):
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
        xr = real root
        fx = function value at root
        ea = approximate relative error (%)
        iter = number of iterations

    -
    '''
    if xl is None or xu is None:
        raise ValueError("At least 3 input arguments required")
    test = func(xl, *args) * func(xu, *args)
    if test > 0:
        raise ValueError("No sign change")
    iter = 0
    xr = xl
    ea = 100
    while True:
        xrold = xr
        xr = (xl + xu) / 2
        iter += 1
        if xr != 0:
            ea = abs((xr - xrold) / xr) * 100
        test = func(xl, *args) * func(xr, *args)
        if test < 0:
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0
        if ea <= es or iter >= maxit:
            break
    fx = func(xr, *args)
    return xr, fx, ea, iter


# In[5]:


def bisect2(func):
    '''
    bisect2:
        uses bisection method to find the root of func
    
    func input:
        func = name of function
        
    input:
        xl, xu = lower and upper guesses
        es = desired relative error
        maxit = maximum allowable iterations
    
    output:
        x = real root
        y = function value at root
    
    -
    '''
    xl = float(input('enter lower bound xl = '))
    xu = float(input('enter upper bound xu = '))
    es = float(input ('allowable tolerance es = '))
    maxit = int(input('maximum number of iteration maxit = '))
    a = [xl]
    b = [xu]
    ya = [func(a[0])]
    yb = [func(b[0])]
    if ya[0] * yb[0] > 0.0:
        raise Exception('Function has same sign at end points')
    x = []
    y = []
    for i in range(maxit):
        x.append((a[i] + b[i]) / 2)
        y.append(func(x[i]))
        if abs(x[i] - a[i]) < es:
            print('\nBisection method has converged\n')
            break
        if y[i] == 0.0:
            print('\nexact zero found\n')
            break
        elif y[i] * ya[i] < 0:
            a.append(a[i])
            ya.append(ya[i])
            b.append(x[i])
            yb.append(y[i])
        else:
            a.append(x[i])
            ya.append(y[i])
            b.append(b[i])
            yb.append(yb[i])
        iter = i+1
    if iter >= maxit:
        print('\nzero not found to desired tolerance\n')
    n = len(x)
    k = list(range(1, n+2))
    out = list(zip(k, a[:n+1], b[:n+1], x, y))
    print('step\t\t\txl\t\t\txu\t\t\txr\t\t\tf(xr)')
    for step, xl, xu, xr, f_xr in out:
        print(f'{step:<24}{xl:<24}{xu:<24}{xr:<24}{f_xr:<24}')
    return x[-1], y[-1]

# In[6]:


def false_position(func):
    '''
    false_position:

    func input:
        func = name of function
    
    input:
        xl, xu = lower and upper guesses
        es = desired relative error
        maxit = maximum allowable iterations

    output:
        x = real root
        y = function value at root

    -
    '''
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
            print('\nFalse position method has converged\n')
            break
    if i >= maxit:
        print('\nzero not found to desired tolerance\n')

    k = list(range(1, i+3))
    out = list(zip(k, a[:i+2], b[:i+2], x, y))
    print('step\t\t\txl\t\t\txu\t\t\txr\t\t\tf(xr)')
    for step, xl, xu, xr, f_xr in out:
        print(f'{step:<24}{xl:<24}{xu:<24}{xr:<24}{f_xr:<24}')

    return x[-1], y[-1]


# In[7]:


def multiple1(func, dfunc):
    '''
    multiple1:

    func input:
        func  = name of function
        dfunc = name of the derivated function

    input:
        m = multiplicity of the root
        xguess = initial guess
        es = desired relative error
        maxit = maximum number of iterations
    
    output:
        x = real root
        f = function value at root
    
    -
    '''
    m = float(input("Enter the multiplicity of the root = "))
    xguess = float(input("Enter the initial guess for x = "))
    es = float(input("Enter the allowable tolerance = "))
    maxit = int(input("Enter the maximum number of iterations = "))
    
    iter = 1
    x = [xguess]
    f = [func(xguess)]
    dfdx = [dfunc(xguess)]
    
    for i in range(1, maxit):
        x_new = x[i-1] - m * f[i-1] / dfdx[i-1]
        x.append(x_new)
        f_new = func(x_new)
        f.append(f_new)
        dfdx_new = dfunc(x_new)
        dfdx.append(dfdx_new)
        
        if abs(x[i] - x[i-1]) < es:
            print("\nNewton's method has converged.\n")
            break
        iter = i+1
    
    if iter >= maxit:
        print("\nZero not found to desired tolerance.\n")
    
    k = list(range(1, iter+2))
    out = list(zip(k, x, f, dfdx))
    print('Step\t\t\tx\t\t\tf\t\t\tdf/dx')
    for step, xr, f_xr, dfdx_xr in out:
        print(f'{step:<24}{xr:<24}{f_xr:<24}{dfdx_xr:<24}')
    return x[-1], f[-1]


# In[8]:


def multiple2(func, dfunc, ddfunc):
    '''
    multiple2:

    func input:
        func  = name of function
        dfunc = name of the derivated function
        ddfunc = name of second order derivative function

    input:
        xguess = initial guess
        es = desired relative error
        maxit = maximum number of iterations

    output:
        x = real root
        f = function value at root
    
    -
    '''
    xguess = float(input('Enter initial guess: xguess = '))
    es = float(input('Allowable tolerance es = '))
    maxit = int(input('Maximum number of iterations: maxit = '))
  
    iter = 1
    x = [0] * (maxit + 1)
    f = [0] * (maxit + 1)
    dfdx = [0] * (maxit + 1)
    d2fdx2 = [0] * (maxit + 1)
  
    x[0] = xguess
    f[0] = func(x[0])
    dfdx[0] = dfunc(x[0])
    d2fdx2[0] = ddfunc(x[0])
  
    for i in range(1, maxit):
        x[i] = x[i - 1] - f[i - 1] * dfdx[i - 1] / (dfdx[i - 1]**2 - f[i - 1] * d2fdx2[i - 1])
        f[i] = func(x[i])
        dfdx[i] = dfunc(x[i])
        d2fdx2[i] = ddfunc(x[i])
        ea = abs(x[i] - x[i - 1])
        if ea < es:

            print('\nNewton method has converged\n')
            break
        iter = i + 1
  
    if iter >= maxit:
        print('\nZero not found to desired tolerance\n')
  
    #print('    step             x            f            df/dx         d2f/dx2')
    #for i in range(iter + 1):
    #    print('{:5.0f}   {:17.14f}   {:20.15f}   {:20.15f}     {:20.15f}'.format(i, x[i], f[i], dfdx[i], d2fdx2[i]))
    k = list(range(1, iter + 2))
    out = list(zip(k, x, f, dfdx, d2fdx2))
    print('Step\t\t\tx\t\t\tf\t\t\tdf/dx\t\t\td2f/dx2')
    for step, xr, f_xr, dfdx_xr, d2fdx2_xr in out:
        print(f'{step:<24}{xr:<24}{f_xr:<24}{dfdx_xr:<24}{d2fdx2_xr:<24}')
    return x[iter], f[iter]

# In[9]:


def newtraph(func, dfunc, xr, es=1e-4, maxit=50):
    '''
    newtraph:
        uses Newton-Raphson method to find the root of func

    input:
        func = name of function
        dfunc = name of derivative of function
        xr = initial guess
        es = desired relative error (default = 1e-4)
        maxit = maximum allowable iterations (default = 50)

    output:
        root = real root
        ea = approximate relative error (%)
        iter = number of iterations

    -
    '''
    iter = 0
    while True:
        xrold = xr
        xr = xr - func(xr) / dfunc(xr)
        iter += 1
        ea = abs((xr - xrold) / xr) * 100 if xr != 0 else 0
        if ea <= es or iter >= maxit:
            break
    root = xr
    return root, ea, iter


# In[10]:


# code2


# In[11]:


def Cubic_LS(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    sx = np.sum(x)
    sx2 = np.sum(x**2)
    sx3 = np.sum(x**3)
    sx4 = np.sum(x**4)
    sx5 = np.sum(x**5)
    sx6 = np.sum(x**6)
    sy = np.sum(y)
    syx = np.sum(x*y)
    syx2 = np.sum(y*x**2)
    syx3 = np.sum(y*x**3)

    A = np.array([[n, sx, sx2, sx3],
                  [sx, sx2, sx3, sx4],
                  [sx2, sx3, sx4, sx5],
                  [sx3, sx4, sx5, sx6]])
    r = np.array([sy, syx, syx2, syx3]).reshape(-1, 1)
    z = np.linalg.lstsq(A, r, rcond=None)[0].flatten()
    a0, a1, a2, a3 = z

    p = a0 + a1 * x + a2 * x**2 + a3 * x**3
    table = np.hstack([x, y, p, y-p])
    err = np.sum((y-p)**2)

    St = np.sum((y - sy/n)**2)
    Sr = err
    Syx = np.sqrt(Sr/(n-3))
    r = np.sqrt((St-Sr)/St)

    x1 = np.min(x)
    x2 = np.max(x)
    xx = np.linspace(x1, x2, num=50)
    yy = a0 + a1 * xx + a2 * xx**2 + a3 * xx**3

    plt.plot(x, y, 'r*', xx, yy, 'b')
    plt.show()


# In[12]:


def Gauss_Newton(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    a = np.array(input("Enter the initial guesses [a0, a1] =").split(), dtype=float)
    tol = float(input("Enter the tolerance to1 = "))
    itmax = int(input("Enter the maximum iteration number itmax = "))

    n = len(x)
    print("     iter      a0      a1      da0       da1")
    for iter in range(itmax):
        a0, a1 = a[0], a[1]
        f = np.exp(-a0 * x) * np.cos(a1 * x)
        d = y - f
        z = np.column_stack((-x * np.exp(-a0 * x) * np.cos(a1 * x), -x * np.exp(-a0 * x) * np.sin(a1 * x)))
        da = np.linalg.inv(z.T @ z) @ (z.T @ d)
        a += da
        out = [iter, a, da]
        print(*out)
        if np.all(np.abs(da) < tol):
            print("Gauss-Newton method has converged")
            break

    x1, x2 = np.min(x), np.max(x)
    xx = np.linspace(x1, x2, 50)
    yy = np.exp(-a0 * xx) * np.cos(a1 * xx)
    plt.plot(xx, yy, x, y, 'ro')
    plt.show()


# In[13]:


def Lagrange_coef(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    c = [0] * n
    for k in range(n):
        d = 1
        for i in range(n):
            if i != k:
                d *= (x[k] - x[i])
            c[k] = y[k] / d
    return c


# In[14]:


def Lagrange_Eval(t, x, c):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    m = len(x)
    p = []
    for i in range(len(t)):
        p_ = 0
        for j in range(m):
            N = 1
            for k in range(m):
                if j != k:
                    N *= (t[i] - x[k])
            p_ += N * c[j]
        p.append(p_)
    return p


# In[15]:


def Linear_LS(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    m = len(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(x * x)
    sxy = sum(x * y)
    a0 = (sxx * sy - sxy * sx) / (m * sxx - sx**2)
    a1 = (m * sxy - sx * sy) / (m * sxx - sx**2)
    table = np.hstack([x, y, a0 + a1 * x, y - a0 - a1 * x])
    print("       x           y      (a0+a1*x)     (y-a0-a1*x)")
    print(table)
    err = sum(table[:, 3]**2)
    s = np.array([a0, a1])
    St = sum((y - sy / m)**2)
    Sr = err
    Syx = np.sqrt(Sr / (m - 2))
    r = np.sqrt((St - Sr) / St)
    plt.plot(x, y, 'r*', xx, yy, 'g')
    plt.show()
    return s


# In[16]:


def linregr(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must be same length")
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x * x)
    sxy = np.sum(x * y)
    sy2 = np.sum(y * y)
    a = np.zeros((2, 1))
    a[0] = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
    a[1] = sy / n - a[0] * sx / n
    r2 = ((n * sxy - sx * sy) / np.sqrt(n * sx2 - sx ** 2) / np.sqrt(n * sy2 - sy ** 2)) ** 2
    xp = np.linspace(np.min(x), np.max(x), 2)
    yp = a[0] * xp + a[1]
    plt.plot(x, y, 'o', xp, yp)
    plt.grid(True)
    plt.show()
    return a, r2


# In[17]:


def Multiple_Linear(x1, x2, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x1)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y.reshape(-1, 1)

    sx1 = x1.sum()
    sx2 = x2.sum()
    sx1x2 = (x1 * x2).sum()
    sx1x1 = (x1**2).sum()
    sx2x2 = (x2**2).sum()
    sy = y.sum()
    sx1y = (x1 * y).sum()
    sx2y = (x2 * y).sum()

    A = np.array([[n, sx1, sx2], [sx1, sx1x1, sx1x2], [sx2, sx1x2, sx2x2]])
    r = np.array([sy, sx1y, sx2y]).reshape(-1, 1)
    z = np.linalg.solve(A, r)
    a0, a1, a2 = z.flatten()

    table = np.hstack([x1, x2, y, (a0 + a1 * x1 + a2 * x2), (y - a0 - a1 * x1 - a2 * x2)])
    err = (table[:, 4]**2).sum()

    St = ((y - sy/n)**2).sum()
    Sr = err
    Syx = np.sqrt(Sr / (n - 3))
    r = np.sqrt((St - Sr) / St)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(x1, x2, y)

    x1a = x1.min()
    x1b = x1.max()
    x1s = np.linspace(x1a, x1b, 50)
    x2a = x2.min()
    x2b = x2.max()
    x2s = np.linspace(x2a, x2b, 50)

    xx1, xx2 = np.meshgrid(x1s, x2s)
    yy = a0 + a1 * xx1 + a2 * xx2
    ax.plot_surface(xx1, xx2, yy)
    plt.show()


# In[18]:


def Newtint2(x, y, xx):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    if len(y) != n:
        raise ValueError('x and y must be same length')
    b = [[0 for j in range(n)] for i in range(n)]
    b[:, 0] = y
    for j in range(1, n):
        for i in range(n-j+1):
            b[i][j] = (b[i+1][j-1] - b[i][j-1]) / (x[i+j-1] - x[i])
    yint = [0 for k in range(len(xx))]
    for k in range(len(xx)):
        xt = 1
        yint[k] = b[0][0]
        for j in range(n-1):
            xt = xt * (xx[k] - x[j])
            yint[k] = yint[k] + b[0][j+1] * xt
    return yint


# In[19]:


def quadratic(x, f):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    h3 = x[3] - x[2]
    h4 = x[4] - x[3]
    f1 = f[0]
    f2 = f[1]
    f3 = f[2]
    f4 = f[3]
    f5 = f[4]
    
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, h1, h1**2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, h2, h2**2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, h3, h3**2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h4, h4**2],
        [0, 1, 2*h1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 2*h2, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 2*h3, 0, -1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    b = np.array([f1, f2-f1, f2, f3-f2, f3, f4-f3, f4, f5-f4, 0, 0, 0, 0])
    
    return A, b


# In[20]:


def Quadratic_LS(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    sx = np.sum(x)
    sx2 = np.sum(x**2)
    sx3 = np.sum(x**3)
    sx4 = np.sum(x**4)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2y = np.sum(x * x * y)
    A = np.array([[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]])
    r = np.array([sy, sxy, sx2y]).reshape(-1, 1)
    z = np.linalg.solve(A, r)
    a0, a1, a2 = z.flatten()
    table = np.hstack((x, y, (a0 + a1 * x + a2 * x**2), (y - a0 - a1 * x - a2 * x**2)))
    print('x     y      (a0+a1*x+a2*x.^2)  (y-a0-a1*x-a2*x.^2)')
    print(table)
    err = np.sum(table[:, 3]**2)
    St = np.sum((y - sy / n)**2)
    Sr = err
    Syx = np.sqrt(Sr / (n - 3))
    r = np.sqrt((St - Sr) / St)
    plt.plot(x, y, 'r*')
    xx = np.linspace(np.min(x), np.max(x), 50)
    yy = a0 + a1 * xx + a2 * xx**2
    plt.plot(xx, yy, 'm')
    plt.show()


# In[21]:


# code3


# In[22]:


def Cholesky(A):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, m = A.shape
    U = np.zeros((n, n))
    for i in range(n):
        U[i,i] = np.sqrt(A[i,i] - np.dot(U[:i, i], U[:i, i]))
        for j in range(i+1, n):
            U[i,j] = (A[i,j] - np.dot(U[:i,i], U[:i,j])) / U[i,i]
    L = np.transpose(U)
    return L, U


# In[23]:


def GaussNaive(A, b):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A must be square")
    nb = n + 1
    Aug = np.column_stack((A, b))
    # forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = Aug[i][k] / Aug[k][k]
            Aug[i, k:nb] -= factor * Aug[k, k:nb]
    # back substitution
    x = np.zeros(n)
    x[n - 1] = Aug[n - 1][nb - 1] / Aug[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (Aug[i][nb - 1] - np.dot(Aug[i, i + 1:n], x[i + 1:n])) / Aug[i][i]
    return x


# In[24]:


def GaussPivot(A, b):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Compute the matrix sizes
    m, n = np.shape(A)
    if m != n:
        raise ValueError("Matrix A must be square")
    nb = n + 1
    Aug = np.c_[A, b]
    # forward elimination
    for k in range(n-1):
        # partial pivoting
        big = np.argmax(np.abs(Aug[k:n, k]))
        ipr = big + k
        if ipr != k:
            Aug[[k, ipr], :] = Aug[[ipr, k], :]
        for i in range(k+1, n):
            factor = Aug[i, k] / Aug[k, k]
            Aug[i, k:nb] = Aug[i, k:nb] - factor * Aug[k, k:nb]
    # back substitution
    x = np.zeros(n)
    x[n-1] = Aug[n-1, nb-1] / Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (Aug[i, nb-1] - np.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]
    return x


# In[25]:


def LU_factor(A, unitdiag=None):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(A[:, 0])
    flag = 0
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    if unitdiag is None:
        lflag = 1
    elif unitdiag == 'L':
        lflag = 1
    else:
        lflag = 0

    if A[0, 0] == 0:
        flag = 1
    
    if flag == 0:
        if lflag == 1:
            L[0, 0] = 1
            U[0, 0] = A[0, 0]
        else:
            U[0, 0] = 1
            L[0, 0] = A[0, 0]
        
        for j in range(1, n):
            U[0, j] = A[0, j] / L[0, 0]
            L[j, 0] = A[j, 0] / U[0, 0]
    
    if flag == 0:
        for i in range(1, n - 1):
            if lflag == 1:
                L[i, i] = 1
                U[i, i] = A[i, i] - np.dot(L[i, :i], U[:i, i])
            else:
                U[i, i] = 1
                L[i, i] = A[i, i] - np.dot(L[i, :i], U[:i, i])
            
            if (U[i, i] == 0 or L[i, i] == 0):
                flag = 1
            
            if flag == 0:
                for j in range(i + 1, n):
                    U[i, j] = (A[i, j] - np.dot(L[i, :i], U[:i, j])) / L[i, i]
                    L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
        
        if flag == 0:
            if lflag == 1:
                L[n - 1, n - 1] = 1
                U[n - 1, n - 1] = A[n - 1, n - 1] - np.dot(L[n - 1, :n - 1], U[:n - 1, n - 1])
            else:
                U[n - 1, n - 1] = 1
                L[n - 1, n - 1] = A[n - 1, n - 1] - np.dot(L[n - 1, :n - 1], U[:n - 1, n - 1])
    
    if flag == 1:
        L = 'Factorization impossible'
        U = ' '
    
    return L, U


# In[62]:


def LU_pivot(A):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, n1 = A.shape
    L = np.eye(n)
    P = np.eye(n)
    U = A
    for j in range(n):
        pivot, m = np.max(np.abs(U[j:n, j])), np.argmax(np.abs(U[j:n, j]))
        m += j - 1
        if m != j:
            U[[j, m], :] = U[[m, j], :]
            P[[j, m], :] = P[[m, j], :]
            if j >= 2:
                L[j, 0:j-1], L[m, 0:j-1] = L[m, 0:j-1].copy(), L[j, 0:j-1].copy()
        for i in range(j+1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, :] = U[i, :] - L[i, j] * U[j, :]
    return L, U, P


# In[27]:


def LU_Solve(L, U, b):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, m = np.shape(L)
    d = np.zeros((n, 1))
    x = np.zeros((n, 1))
    # solve L d =b using forward substitution
    d[0][0] = b[0][0] / L[0][0]
    for i in range(1, n):
        d[i][0] = (b[i][0] - np.dot(L[i][0:i], d[0:i])) / L[i][i]
    # Solve U x = d using back substitution
    x[n-1][0] = d[n-1][0] / U[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i][0] = (d[i][0] - np.dot(U[i][i+1:n], x[i+1:n])) / U[i][i]
    return x


# In[28]:


def Tridiag(e, f, g, r):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(f)
    for k in range(1, n):
        factor = e[k] / f[k-1]
        f[k] = f[k] - factor * g[k-1]
        r[k] = r[k] - factor * r[k-1]
    x = [0] * n
    x[n-1] = r[n-1] / f[n-1]
    for k in range(n-2, -1, -1):
        x[k] = (r[k] - g[k] * x[k+1]) / f[k]
    return x


# In[29]:


def Truss(alpha, beta, gamma, delta):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    A = np.zeros((10, 10))
    A[0, 0] = 1
    A[0, 4] = np.sin(alpha)
    A[1, 1] = 1
    A[1, 3] = 1
    A[1, 4] = np.cos(alpha)
    A[2, 6] = np.sin(beta)
    A[2, 7] = np.sin(gamma)
    A[3, 3] = -1
    A[3, 5] = 1
    A[3, 6] = -np.cos(beta)
    A[3, 7] = np.cos(gamma)
    A[4, 2] = 1
    A[4, 8] = np.sin(gamma)
    A[5, 5] = -1
    A[5, 8] = -np.cos(delta)
    A[6, 4] = -np.sin(alpha)
    A[6, 6] = -np.sin(beta)
    A[7, 4] = -np.cos(alpha)
    A[7, 6] = np.cos(beta)
    A[7, 9] = 1
    A[8, 7] = -np.sin(gamma)
    A[8, 8] = -np.sin(delta)
    A[9, 7] = -np.cos(gamma)
    A[9, 8] = np.cos(delta)
    A[9, 9] = -1
    
    b = np.zeros((10, 1))
    b[2, 0] = 100
    f = np.linalg.solve(A, b)
    
    return A, b, f


# In[30]:


# code4


# In[31]:


def fixed_pt_sys(G, x0, es=1e-5, maxit=50):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Solve the nonlinear system x = G(x) using fixed-point method
    # Vectors x and x0 are row vectors (for display purposes)
    # function G returns a column vector, [g1(x), ..gn(x)]'
    # stop if norm of change in solution vector is less than es
    # y = feval(G,xold); the next approximate solution is xnew = y';

    print(np.array([0, x0])) # display initial estimate
    xold = x0
    iter = 1
    while (iter <= maxit):
        y = G(xold)
        xnew = y.T
        dif = np.linalg.norm(xnew - xold)
        print(np.array([iter, xnew, dif]))
        if dif <= es:
            x = xnew
            print("Fixed-point iteration has converged")
            return x
        else:
            xold = xnew
        iter = iter + 1
    print("Fixed-point iteration did not converge")
    x = xnew
    return x


# In[32]:


def GaussSeidel(A, b, es=0.00001, maxit=50):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A must be square")
    C = np.copy(A)
    for i in range(n):
        C[i][i] = 0
        x[i] = 0
    x = np.transpose([x])
    for i in range(n):
        C[i] /= A[i][i]
    d = np.divide(b, np.diagonal(A))
    iter = 0
    print("     i       x1      x2      x3      x4       x5...")
    while True:
        xold = np.copy(x)
        for i in range(n):
            x[i] = d[i] - np.dot(C[i], x)
            if x[i] != 0:
                ea[i] = abs((x[i] - xold[i]) / x[i]) * 100
        out = np.insert(x, 0, [iter + 1], axis=1)
        print(out)
        iter += 1
        if max(ea) <= es or iter >= maxit:
            break
    if iter >= maxit:
        print("Gauss Seidel method did not converge")
        print("results after maximum number of iterations")
    else:
        print("Gauss Serdel mehtod has converged")
    return x


# In[33]:


def InvPower(A, max_it, tol):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, nn = A.shape
    z = np.ones((n,1))
    it = 0
    error = 100
    L, U = LU_factor(A)
    
    while (it < max_it and error > tol):
        w = LU_Solve(L, U, z)
        ww = np.abs(w)
        kk = np.argmax(ww) # index of max element of ww
        m = (z.T @ z) / (z.T @ w) # estimate of eigenvalue
        z = w / w[kk] # estimate of eigenvector
        out = [it+1, m, z.T]
        print(out)
        error = np.linalg.norm(A @ z - m * z)
        it = it + 1
    
    return z, m, error


# In[34]:


def LU_Solve_Gen(L, U, B):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, n2 = L.shape
    m1, m = B.shape
    
    # Solve Ld = B using forword substritution
    d = np.zeros((n, m))
    for j in range(m):
        d[0, j] = B[0, j]
        for i in range(1, n):
            d[i, j] = B[i, j] - np.dot(L[i, 0:i-1], d[0:i-1, j])
    
    # Solve Ux = d using back substitution
    x = np.zeros((n, m))
    for j in range(m):
        x[n-1, j] = d[n-1, j] / U[n-1, n-1]
        for i in range(n-2, -1, -1):
            x[i, j] = (d[i, j] - np.dot(U[i, i+1:n], x[i+1:n, j])) / U[i, i]
    
    return x


# In[35]:


def Newton_sys(F, JF, x0, tol, maxit):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    xold = x0
    print([0, xold])
    iter = 1
    while (iter <= maxit):
        y = np.linalg.solve(-feval(JF, xold), feval(F, xold))
        xnew = xold + y
        dif = np.linalg.norm(xnew - xold)
        print([iter, xnew, dif])
        if dif <= tol:
            x = xnew
            print('Newton method has converged')
            return x
        else:
            xold = xnew
        iter = iter + 1
    print('Newton method did not converge')
    x = xnew
    return x


# In[36]:


def Power_eig(A, max_it, tol):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, nn = A.shape
    z = np.ones((n,1))
    it = 0
    error = 100
    print('      it         m       z(1)      z(2)       z(3)      z(4)      z(5)')
    while (it < max_it and error > tol):
        w = np.dot(A, z)
        ww = np.abs(w)
        k = np.amax(ww)
        kk = np.argmax(ww)
        m = w[kk]
        z = w / w[kk]
        out = [it + 1, m, z.T]
        print(out)
        error = np.linalg.norm(np.dot(A, z) - m * z)
        it = it + 1
    return error


# In[37]:


def SOR(A, b, x0, w, tol, max_it):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n, m = A.shape
    x = x0.copy()
    C = -A.copy()
    for i in range(n):
        C[i, i] = 0
    for i in range(n):
        C[i, :] = C[i, :] / A[i, i]
    for i in range(n):
        r[i, 0] = b[i] / A[i, i]
    i = 1
    print('    i     x1      x2       x3     ....')
    while (i <= max_it):
        xold = x.copy()
        for j in range(n):
            x[j] = (1 - w) * xold[j] + w * (np.dot(C[j, :], x) + r[j, 0])
        if np.linalg.norm(xold - x) <= tol:
            print('SOR method converged')
            return x
        print([i, x])
        i += 1
    print('SOR method did not converge')
    return x


# In[38]:


# code5


# In[39]:


def Gauss_quad(f, a, b, k):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    t = np.array([
        [-0.5773502692, -0.7745966692, -0.8611363116, -0.9061798459],
        [0.5773502692, 0, -0.3399810436, -0.5384693101],
        [0, 0.7745966692, 0.3399810436, 0],
        [0, 0, 0.8611363116, 0.5384693101],
        [0, 0, 0, 0.9061798459]
    ])

    c = np.array([
        [1, 0.5555555556, 0.3478548451, 0.2369268850],
        [1, 0.8888888889, 0.6521451549, 0.4786286705],
        [0, 0.5555555556, 0.6521451549, 0.5688888889],
        [0, 0, 0.3478548451, 0.4786286705],
        [0, 0, 0, 0.2369268850]
    ])

    x = 0.5 * ((b - a) * t[:k, k - 1] + b + a)
    y = f(x)
    tt = t[:k, k - 1]
    cc = c[:k, k - 1]
    cd = cc.transpose()
    int = np.dot(y, cd)
    I = int * (b - a) / 2

    return I


# In[40]:


def quadstep(f, a, b, tol, fa, fc, fb, *args):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Recursive subfunction used by quadadapt.
    h = b - a
    c = (a + b) / 2
    fd = f((a + c) / 2, *args)
    fe = f((c + b) / 2, *args)
    q1 = h / 6 * (fa + 4 * fc + fb)
    q2 = h / 12 * (fa + 4 * fd + 2 * fc + 4 * fe + fb)
    if abs(q2 - q1) <= tol:
        q = q2 + (q2 - q1) / 15
    else:
        qa = quadstep(f, a, c, tol, fa, fd, fc, *args)
        qb = quadstep(f, c, b, tol, fc, fe, fb, *args)
        q = qa + qb
    return q


# In[41]:


def quadadapt(f, a, b, tol, *args):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Evaluates definite integral of f(x) from a to b
    if tol is None:
        tol = 1e-6
    c = (a + b) / 2
    fa = f(a, *args)
    fc = f(c, *args)
    fb = f(b, *args)
    q = quadstep(f, a, b, tol, fa, fc, fb, *args)
    return q


# In[42]:


def romberg(func, a, b, es=0.000001, maxit=50, *args):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Romberg integration
    # Input:
    #   func: name of function to be integrated
    #   a, b: integration limits
    #   es: desired relative error (default = 0.000001%)
    #   maxit: maximum allowable iterations (default = 30)
    #   args: additional parameters used by func
    # Output:
    #   q: integral estimate
    #   ea: approximate relative error (%)
    #   iter: number of iterations

    if es is None:
        es = 0.000001
    if maxit is None:
        maxit = 50

    n = 1
    I = [[0 for i in range(maxit + 1)] for j in range(maxit + 1)]
    I[0][0] = trap(func, a, b, n, *args)
    iter = 0

    while iter < maxit:
        iter += 1
        n = 2 ** iter
        I[iter][0] = trap(func, a, b, n, *args)
        for k in range(1, iter + 1):
            j = 1 + iter - k
            I[j - 1][k] = (4 ** (k - 1) * I[j][k - 1] - I[j - 1][k - 1]) / (4 ** (k - 1) - 1)
        ea = abs((I[0][iter] - I[1][iter - 1]) / I[0][iter]) * 100
        if ea <= es:
            break

    q = I[0][iter]
    return q, ea, iter


# In[43]:


def Simp(f, a, b, n):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # integral of f using composite Simpson rule
    # n must be even
    h = (b - a) / n
    S = f(a)
    x = np.zeros(n)
    for i in range(0, n-1, 2):
        x[i] = a + h*i
        S += 4 * f(x[i])
    for i in range(2, n-2, 2):
        x[i] = a + h*i
        S += 2 * f(x[i])
    S += f(b)
    I = h * S / 3
    return I


# In[44]:


def trap(func, a, b, n=100, *args):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    if b <= a:
        raise ValueError("Upper bound must be greater than lower")
    x = a
    h = (b - a) / n
    s = func(a, *args)
    for i in range(1, n):
        x = x + h
        s = s + 2 * func(x, *args)
    s = s + func(b, *args)
    return (b - a) * s / (2 * n)


# In[45]:


def trapuneq(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must be the same length")
    s = 0
    for i in range(n - 1):
        if x[i + 1] < x[i]:
            raise ValueError("x values must be in ascending order")
        s += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    return s


# In[46]:


# code6


# In[47]:


def Euler_sys(f, tspan, y0, h):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    a = tspan[0]
    b = tspan[1]
    n = int((b-a)/h)
    t = [a + i*h for i in range(1, n+1)]
    k = f(a, y0)
    y = [y0 + h * k]
    for i in range(n-1):
        k = f(t[i], y[i])
        y.append(y[i] + h * k)
    t = [a] + t
    y = [y0] + y
    out = list(zip(t, y))
    print("     t           y1        y2       y3  ...")
    for i in out:
        print("{:8.3f}  {:15.10f}  {:15.10f}".format(*i))


# In[48]:


def eulode(dydt, tspan, y0, h, *args):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # eulode: Euler ODE solver
    #   [t,y] = eulode(dydt,tspan,y0,h,p1,p2,...):
    #           uses Euler's method to integrate an ODE
    # input:
    #   dydt = name of the M-file that evaluates the ODE
    #   tspan = [ti, tf] where ti and tf = initial and
    #   final values of independent variable
    #   y0 = initial value of dependent variable
    #   h = step size
    #   p1,p2,... = additional parameters used by dydt
    # output:
    #   t = vector of independent variable
    #   y = vector of solution for dependent variable
    if len(args) < 4:
        raise Exception("at least 4 input arguments required")

    ti = tspan[0]
    tf = tspan[1]
    if tf <= ti:
        raise Exception("upper limit must be greater than lower")

    t = np.arange(ti, tf + h, h)
    n = len(t)

    # if necessary, add an additional value of t
    # so that range goes from t = ti to tf
    if t[-1] < tf:
        t = np.append(t, tf)
        n = n + 1

    y = np.ones(n) * y0 #preallocate y to improve efficiency
    for i in range(n - 1): #implement Euler's method
        y[i + 1] = y[i] + dydt(t[i], y[i], *args) * (t[i + 1] - t[i])

    print("    step        t             y")
    k = np.arange(1, len(t) + 1)
    out = np.vstack((k, t, y))
    print(f"{out[0, i]:5d}    {out[1, i]:15.10f}   {out[2, i]:15.10f}")

    return t, y


# In[49]:


def example2_e(t):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    return 1.5 * np.exp(-t) + 0.5 * np.sin(t) - 0.5 * np.cos(t)


# In[50]:


def example2_f(t, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    return np.sin(t) - y


# In[51]:


def example3(t, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    return t * y**0.5


# In[52]:


def example5(t, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    f1 = -0.5 * y[0]
    f2 = 4 - 0.1 * y[0] - 0.3 * y[1]
    f = [f1, f2]
    return f


# In[53]:


def Heun_iter(f, tspan, y0, h, itmax):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # solve y' = f(t,y) with intitial condition y(a) = y0
    # using n steps of the Heun's method;
    # itmax = 0: Heun's mehtod without iterative correction

    a = tspan[0]
    b = tspan[1]
    n = int((b-a)/h)
    t = [a+h*i for i in range(1,n+1)]
    k1 = f(a, y0)
    k2 = f(a + h, y0 + k1 * h)
    y = [y0 + 0.5 * (k1 + k2) * h]
    for iter in range(itmax):
        k2 = f(a + h, y[0])
        y[0] = y0 + 0.5 * (k1 + k2) * h
    for i in range(n-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + k1 * h)
        y.append(y[i] + 0.5 * (k1 + k2) * h)
        for iter in range(itmax):
            k2 = f(t[i] + h, y[i + 1])
            y[i + 1] = y[i] + 0.5 * (k1 + k2) * h
    t = [a] + t
    y = [y0] + y
    return t, y


# In[54]:


def midpoint(f, tspan, y0, h):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    a, b = tspan
    n = int((b-a)/h)
    t = np.arange(a+h, b+h, h)
    k1 = f(a, y0)
    k2 = f(a + h/2, y0 + k1/2*h)
    y = np.zeros(n)
    y[0] = y0 + k2*h
    for i in range(n-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + k1/2*h)
        y[i+1] = y[i] + k2*h
    t = np.concatenate(([a], t))
    y = np.concatenate(([y0], y))
    print("    step        t           y")
    for i, (time, value) in enumerate(zip(t, y)):
        print("{:5d}  {:15.10f}  {:15.10f}".format(i, time, value))
    return t, y


# In[55]:


def RK4(f, tspan, y0, h):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    a, b = tspan[0], tspan[1]
    n = int((b-a)/h)
    t = np.arange(a+h, b+h, h)
    k1 = f(a, y0)
    k2 = f(a + h/2, y0 + k1/2*h)
    k3 = f(a + h/2, y0 + k2/2*h)
    k4 = f(a + h, y0 + k3*h)
    y = np.zeros(n+1)
    y[0] = y0 + (k1/6 + k2/3 + k3/3 + k4/6)*h
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + k1/2*h)
        k3 = f(t[i] + h/2, y[i] + k2/2*h)
        k4 = f(t[i] + h, y[i] + k3*h)
        y[i+1] = y[i] + (k1/6 + k2/3 + k3/3 + k4/6)*h
    t = np.append(a, t)
    y = np.append(y0, y)
    print('step\t\tt\t\ty')
    for i in range(len(t)):
        print(f'{i}\t\t{t[i]:.10f}\t{y[i]:.10f}')


# In[56]:


def RK4_sys(f, tspan, y0, h):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    a, b = tspan
    n = int((b-a)/h)
    t = np.arange(a+h, b+h, h)
    k1 = f(a, y0)
    k2 = f(a + h/2, y0 + k1/2*h)
    k3 = f(a + h/2, y0 + k2/2*h)
    k4 = f(a + h, y0 + k3*h)
    y = np.zeros((n+1, len(y0)))
    y[0, :] = y0 + (k1/6 + k2/3 + k3/3 + k4/6)*h
    for i in range(n):
        k1 = f(t[i], y[i, :])
        k2 = f(t[i] + h/2, y[i, :] + k1/2*h)
        k3 = f(t[i] + h/2, y[i, :] + k2/2*h)
        k4 = f(t[i] + h, y[i, :] + k3*h)
        y[i+1, :] = y[i, :] + (k1/6 + k2/3 + k3/3 + k4/6)*h
    t = np.concatenate([[a], t])
    y = np.concatenate([np.expand_dims(y0, axis=0), y], axis=0)
    print("t\ty1\ty2\t...")
    for i in range(len(t)):
        print("{:.3f}\t{:.10f}".format(t[i], *y[i, :]))


# In[57]:


# code7


# In[58]:


def example1_f(x, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    htc = 0.01
    Ta = 20
    f1 = y[1]
    f2 = -htc * (Ta - y[0])
    return [f1, f2]


# In[59]:


def example2_f(t, y):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    htc = 5 * 10**(-8)
    Ta = 20
    f1 = y[1]
    f2 = -htc * (Ta**4 - y[0]**4)
    return [f1, f2]


# In[60]:


def linear_FD():
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # script for finite-difference ODE-BVP
    # y_xx = p(x) y_x +q(x) y +r(x)    aa<=x<=bb
    # y(aa) = ya; y(bb) = yb
    # ************************* begin problem definition

    aa = float(input('left boundary aa = '))
    bb = float(input('right boundary bb = '))
    n = int(input('number of subintervals n = '))
    # boundary conditions
    ya = float(input('left boundary condition ya = '))
    yb = float(input('right boundary condition yb = '))
    # define p(x), q(x), r(x)
    # Note: use p(x) ='c*x.^0' if p(x) = constant,
    #           p(x) ='0*x' if p(x) =0
    px= input('function p(x) = ')
    qx= input('function q(x) = ')
    rx= input('function r(x) = ')

    h = (bb-aa)/n
    h2 = h/2
    hh = h*h
    x = numpy.linspace(aa+h, bb, n)
    pp = eval(px)
    qq = eval(qx)
    rr = eval(rx)
    p = pp[:n-1]
    q = qq[:n-1]
    r = rr[:n-1]
    # upper diagonal (a), diagonal (d), lower diagonal (b)
    a = numpy.zeros(n-1)
    b = a
    a[1:n-1] = 1 - p[1:n-1]*h2
    d = -(2+hh*q)
    b[:n-2] = 1 + p[:n-2]*h2
    c = numpy.zeros(n-1)
    c[0] = hh*r[0] - (1+p[0]*h2)*ya
    c[1:n-2] = hh*r[1:n-2]
    c[n-2] = hh*r[n-2]-(1-p[n-2]*h2)*yb

    y = Tridiag(a, d, b, c)
    xx = numpy.concatenate([[aa], x])
    yy = numpy.concatenate([[ya], y, [yb]])
    out = numpy.column_stack([xx, yy])
    print(out)
    plt.plot(xx,yy,'b-o')
    plt.show()


# In[61]:


def shoot_secant(fun, tspan, ya, g, t1, t2, max_it, tol):
    '''
    false_position:


    input:


    output:

        
    -
    '''
    # Nonlinear shooting method based on secant method
    # convert BVP y''=f(x,y,y'); y(a) = ya; g(y(b),y'(b)) = 0
    # into IVP    u''=f(x,u,u'); u(a) = ya; u'(a) = t
    # update t by secant rule to find zero of the error function:
    # m(t) = g(u(b),u'(b)) - residual at x = b
    # stop when abs(m(t)) < tol or after max_it iterations
    
    t = np.zeros(max_it)
    t[0] = t1
    t[1] = t2
    test = 1
    i = 1

    while (test > tol) and (i <= max_it):
        if i > 2:
            t[i] = t[i-1] - (t[i-1]-t[i-2])*m[i-1]/(m[i-1] -m[i-2])

        z0 = [ya, t[i]]
        x, z = odeint(fun, tspan, z0)
        n = z.shape[0]
        z1 = z[n-1,0]
        z2 = z[n-1, 1]
        m[i] = z1-g
        test = np.abs(m[i])
        i = i+1
    return t

