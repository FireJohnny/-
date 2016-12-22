#coding:utf-8
__author__ = 'lenovo'

import numpy as np

def f(x):
    return np.array([100 *(x[0]**2-x[1])**2 + (x[0]-1)**2])
def grad(x):
    return np.array([[400*x[0]*(x[0]**2 - x[1])+2*(x[0]-1)],[-200*(x[0]**2 -x[1] )]])

def gradAscent(fun,gfun,x):

    #a = fun(x)
    sigma = 0.4
    epsilon = 10**(-5)
    rho = 0.6

    k = 0
    iters = 0
    n = np.shape(x)[0]
    maxIter  = 1000
    while( k < maxIter ):
        gk = gfun(x)
        iters +=1
        iters %= n
        if iters ==1:
            dk = -gk
            #print dk

        else:
            beta = 1.0 * np.dot(gk.T, gk)/np.dot(g0.T, g0)
            #print beta
            dk = -gk + np.dot(beta[0][0], d0)
            gd = gk.T * dk

            if gd.all() >= 0.0:
                dk = -gk
        if np.linalg.norm(gk) < epsilon : break
        m = 0
        mk = 0
        while m < 30:
            d = x + rho**m*dk.T
            f = np.dot(gk.T, dk)
            #print f

            h = sigma*rho ** m * f[0]
            e = fun(x)

            a = fun(d[0])
            b = e + h
            if a <= b:
                mk = m
                break
            m += 1
        x = x + rho ** mk * dk.T
        x= x.tolist()[0]
        #val = fun(x)
        g0 = gk
        d0 = dk
        k += 1
    xk = x
    val = fun(xk)
    return xk, val

if __name__ == "__main__":
    x = [0.0, 0.0]
    #print f(x)
    xk , val =  gradAscent(f,grad, x)
    print xk ,val