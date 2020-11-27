import pandas as pd
import pathlib 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import fftpack
import sympy
import sys
import statsmodels.api as sm
import math

'''cal least squaer method'''
def lsm(y,u,n,m,data_size):
    # make matrix
    omega = np.zros((data_size,n+m))
    for i in range(data_size):
        for j in range(n):
            omega[i,j] = y[ n-j+data_size ]
        for j in range(m):
            omega[i,n+j] = u[ m-j+data_size ]    
    al0 = np.dot( omega.T,omega )
    al1 = np.linalg.inv( al0 )
    al2 = np.dot( al1, omega.T )
    alpha = np.dot( al2,y )
    a  = alpha[:n]
    b  = alpha[(n+1):]
    return a,b

'''linear Kalman filter'''
def kalman_linear(y, xhat, p, f, b, c, v, r, bu, u, n):
    #estimate step  
    xhatm = np.zeros((n, 1))
    pm    = np.zeros((n, n))
    xhatm = np.dot(f, xhat) + np.dot(bu,u)
    pm0   = np.dot(np.dot(b, v), b.T)
    pm    = np.dot(np.dot(f, p), f.T) + pm0
    #filteling step
    yy    = y - np.dot(c, xhatm)
    #r0    = np.eye(n,n) * r[0]
    ss    = np.dot(c, pm)
    S     = np.dot(ss, c.T) + r
    K     = np.dot(np.dot(pm, c.T), np.linalg.inv(S))
    xhat  = xhatm + np.dot(K, yy)
    p     = np.dot((e - np.dot(K, c)), p)
    return xhat, p

'''make jointed array'''
def array_joint(a,b):
    a.append(b)
    return a

