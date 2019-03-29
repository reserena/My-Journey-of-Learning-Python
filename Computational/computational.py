# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:52:00 2019

@author: Serena
"""

'''Assembly of all the function ever written for the projects in 
Computational Finance MGMTMFE405 course'''

#-------------------------most used------------------------------------------#
import numpy as np

def uniform(size, seed, a = 7**5, b = 0, m = 2**23-1):
    '''return a random variables u~u[0,1]
    The defaul a, b, and b is using LGM method parameters'''
    
    
    if (seed == 0) and (b==0):
        print("Input error!")
        return None
    
    if size <= 0:
        print("Invalid size")
        return None
    
    x_n = np.zeros((size+1,))
    x_n[0] = seed
    
    for i in range(1, size+1):
        x_n[i] = (a*x_n[i-1]+b)%m
        
    return x_n[1:]/m

def normal(size, uniform):
    '''Box-Muller method to generate normal variables'''
    
    if len(uniform) < size:
        print('Insufficient uniform variables')
        return None
    
    try:
        u = np.array(uniform)
        u = u.reshape((int(size/2), 2))
    except:
        print('Cannot reshape uniform variables, must be np.darray type')
        return None
    
    zz = np.zeros((int(size/2), 2))
    zz[:,0] = np.sqrt(-2*np.log(u[:,0]))*np.cos(2*np.pi*u[:,1])
    zz[:,1] = np.sqrt(-2*np.log(u[:,0]))*np.sin(2*np.pi*u[:,1])
    
    return zz.reshape((size,))
