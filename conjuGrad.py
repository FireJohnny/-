#coding:utf-8
__author__ = 'FireJohnny'

import numpy as np


def f(x):
    return np.array([100 *(x[0]**2-x[1])**2 + (x[0]-1)**2])
def grad(x):
    return np.array([[400*x[0]*(x[0]**2 - x[1])+2*(x[0]-1)],[-200*(x[0]**2 -x[1])]])

def frcg(fun,gfun,x0):
    #用FR共轭梯度法求解无约束问题
    #x0是初始点，fun和gfun分别是目标函数和梯度
    #x,val分别是近似最优点和最优值，k是迭代次数
    maxk = 5000
    rho = 0.6
    sigma = 0.4
    k = 0
    epsilon = 1e-5
    n = np.shape(x0)[0]
    itern = 0
    while k < maxk:
        gk = gfun(x0)
        itern += 1
        itern %= n
        if itern == 1:
            dk = -gk
            print dk
        else:
            beta = 1.0 * np.dot(gk.T,gk)/np.dot(g0.T,g0)

            dk = -gk + beta*d0
            gd = np.dot(gk.T,dk)
            if gd >= 0.0:
                dk = -gk
        if np.linalg.norm(gk) < epsilon:
            break
        m = 0
        mk = 0
        a=x0 + rho ** m * dk
        b= fun(x0) + sigma * rho ** m * np.dot(dk,gk.T)
        print a,"\n",b
        while m < 20:
            if fun(x0 + rho ** m * dk) <= fun(x0) + sigma * rho ** m * np.dot(dk,gk.T):
                mk = m
                break
            a=x0 + rho ** m * dk
            b= fun(x0) + sigma * rho ** m * np.dot(dk,gk.T)
            print a,b
            m += 1
        x0 += rho**mk*dk
        g0 = gk
        d0 = dk
        k += 1

    return x0,fun(x0),k

if __name__ == "__main__":
    x0 = [0.0, 0.0]
    x0 , xk, k = frcg(f,grad,x0)
    print x0, xk, k