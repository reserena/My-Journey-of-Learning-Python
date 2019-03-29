# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:10:02 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##-------------------------New Functions------------------------------------##

class rates:
    '''Simulate the rate paths using Vasicek or CIR model'''
    
    def __init__(self, m, r0, rbar, sigma, k, z, t, dt, method = 'vasicek', 
                 original = False):
        
        self.m = m
        self.r = r0
        self.rbar = rbar
        self.v = sigma
        self.k = k
        self.z = z
        self.t = t
        self.dt = dt
        self.method = method
        self.original = original ##if True, return the whole paths of rates
        
        
    def model(self):
        
        n = int(self.t/self.dt)
        
        para = np.array([self.m, self.rbar, self.v, self.k, self.t, self.dt])
        
        if (para <= 0).any():
            print("Invalid negative input")
            return None
        
        self.m = int(self.m)
        
        try:
            zmodel = np.array(self.z[:self.m*n]).reshape((self.m, n))
        except:
            print("Insufficent or invalid normal variable")
            return None
        
        rmodel = np.zeros((self.m, n+1)) + self.r
        
        
        if self.method == 'vasicek':
            return self.__vasicek(n, rmodel, zmodel)
        elif self.method == 'cir':
            return self.__cir(n, rmodel, zmodel)
        else:
            print('Invalid method argument')
            return None
    
    def __vasicek(self, n, rmodel, zmodel):
        
        for c in range(1, n+1):
            rrr = rmodel[:,c-1] + self.k*(self.rbar - \
                  rmodel[:,c-1])*self.dt +self.v*np.sqrt(self.dt)*zmodel[:,c-1]
            #rrr[rrr < 0] = 0 ##deal with negative r
            rmodel[:,c] = rrr
             
        if self.original == False:
            return np.mean(np.exp(-self.dt*np.sum(rmodel[:,1:],axis = 1)))
        elif self.original == True:
            return rmodel
        else:
            print('Invalid original argument')
            return None
        
    def __cir(self, n, rmodel, zmodel):
        
        for c in range(1, n+1):
             rrr= rmodel[:,c-1] + self.k*(self.rbar - \
                  rmodel[:,c-1])*self.dt +self.v*np.sqrt(rmodel[:,c-1]*self.dt)*zmodel[:,c-1]
             rrr[rrr < 0] = 0 ##deal with negative r
             rmodel[:,c] = rrr
             
        if self.original == False:
            return np.mean(np.exp(-self.dt*np.sum(rmodel[:,1:],axis = 1)))
        elif self.original == True:
            return rmodel
        else:
            print('Invalid original argument')
            return None

class ZCB:
    '''Closed-form solution for solving the price of zero-coupon bond 
    (pure discount bond) with face value $1 using Vasicek or CIR model'''
    
    def __init__(self, r, rbar, k, sigma, T, t = 0, method = 'Vasicek'):
        
        self.r = r
        self.rbar = rbar
        self.k = k
        self.v = sigma
        self.T = T #when the bond matures
        self.t = t #when the bond starts
        self.method = method
        
    def price(self):
        
        try:
            self.r = np.array(self.r)
        except:
            print("Error in interest rate r argument")
            return None
        
        try:
            self.T = np.array(self.T)
        except:
            print("Error in interest rate T argument")
            return None           
            
        para = np.array([self.rbar, self.k, self.v, self.t])
        
        if (para < 0).any():
            print("Invalid negative variable")
            return None
        
        
        if (self.t >= self.T).any():
            print('Duration of bond is invalid')
            return None
        
        if self.method == 'Vasicek':
            return self.__vasicek()
    
    def __vasicek(self):
        
        bbb = (1/self.k)*(1-np.exp(-self.k*(self.T - self.t)))
        aaa = np.exp((self.rbar - (self.v**2)/(2*(self.k**2)))*(bbb - (self.T - self.t)) - \
                     (self.v**2)/(4*self.k)*(bbb**2))
        
        return aaa*np.exp(-bbb*self.r)    

    def __cir(self):
        
        h1 = np.sqrt(self.k**2 + 2*(self.v**2))
        h2 = (self.k + h1)/2
        h3 = (2*self.k*self.rbar)/(self.v**2)

        bbb = (np.exp(h1*(self.T - self.t))-1)/(h2*(np.exp(h1*(self.T-self.t))-1) + h1)
        aaa = ((h1*np.exp(h2*(self.T - self.t)))/(h2*(np.exp(h1*(self.T-self.t))-1) + h1))**h3
        
        return aaa*np.exp(-bbb*self.r)
            

def G2plusplus(m, x0, y0, r0, phi0, rho, a, b, sigma, eta, t, dt, z, outputr = True):
    
    para = np.array([m, sigma, t, dt])
    
    if (para < 0).any():
        print('Invalid negative arguments')
        return None
    
    if (dt > t):
        print('Invalid time interval dt')
        return None
    
    n = int(t/dt)
    
    if len(z) < m*n*2:
        print('Insufficient normal random variables, z')
        return None
       
    xmodel = np.zeros((m, n+1)) + x0
    ymodel = np.zeros((m, n+1)) + y0
    w1model = np.sqrt(dt)*np.array(z[:int(m*n)]).reshape((m, n))
    w2model = np.sqrt(dt)*(rho*np.array(z[:int(m*n)]).reshape((m, n)) + \
                      np.sqrt(1-rho**2)*np.array(z[int(m*n):]).reshape((m, n)))
    
    rmodel = np.zeros((m, n+1))
    rmodel[:,0] = r0
    for c in range(1, n+1):
        xxx = xmodel[:, c-1] - a*xmodel[:, c-1]*dt + sigma*w1model[:,c-1]
        xmodel[:, c] = xxx#np.maximum(xxx, -phi/2) ##prevent negative r
        yyy = ymodel[:, c-1] - b*ymodel[:, c-1]*dt + eta*w2model[:,c-1]          
        ymodel[:, c] = yyy#np.maximum(yyy, -phi/2) ##prevent negative r
        rrr = xmodel[:, c] + ymodel[:, c] + phi0
        rmodel[:, c] = rrr ##prevent negative r
    
    if outputr == True:
        return rmodel
    else:
        return xmodel, ymodel


##------------------------script zone starts--------------------------------##

if __name__ == '__main__':
    
    m = 10000 ##Simulations
    dt = 1/252 #each time step is a day (assume 252 trading days)
    
    '''Question 1: Vasicek Model'''
    r1 = 0.05
    v1 = 0.18
    k1 = 0.82
    rbar1 = 0.05

    
    '''1(a) Pure Discount Bond'''
    t1a = 0.5
    f1a = 10000
    n1a = int(t1a/dt)
    
    np.random.seed(3)
    z1 = np.random.normal(size = n1a*m)
    
    price1a = f1a*rates(m, r1, rbar1, v1, k1, z1, t1a, dt).model()
    price1a
    
    '''1(b) Coupon Paying Bond'''
    c1b = np.array([30,30,30,30,30,30,30,1030])
    t1b = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    
    n1b = (t1b/dt).astype(int)
    
    np.random.seed(100)
    z2 = np.random.normal(size = int(n1b[-1])*m)

    #All rate paths
    rmodel1b = rates(m, r1, rbar1, v1, k1, z2, t1b[-1], dt, original = True).model()
    
    #Add each PV of cashflows together
    price1b = 0
    for i in range(0, len(c1b)):
        price1b += c1b[i]*np.exp(-dt*np.sum(rmodel1b[:,1:int(n1b[i]+1)], axis = 1))
    price1b = np.mean(price1b)
#    price1b
    
    '''1(c) European Call on Pure Discount Bond'''
    t1c = 3/12 #expires in 3 months
    x1c = 9800 #strike price
    
    rmodel1c = rates(m, r1, rbar1, v1, k1, z1, t1c, dt, original = True).model()
    
    pure1c = f1a*ZCB(rmodel1c[:,-1],rbar1, k1, v1, t1a, t1c).price()
    rrr = np.exp(-dt*np.sum(rmodel1c[:,1:], axis = 1))
    
    payoff1c = pure1c - x1c
    payoff1c[payoff1c < 0] = 0
    call1c = np.mean(rrr*payoff1c)
    
    '''1(d) European Call on Coupon Paying Bond'''
    #same call features as 1c
    
    #solve for r*
    r_starr = np.array([0.14, 0.15])
    r_mid = np.mean(r_starr)
    bond1d = np.sum(c1b*ZCB(r_mid, rbar1, k1, v1, t1b, t1c).price()) 
    i = 0
    while round(bond1d,6) != x1c:
        
        if bond1d > x1c:
            r_starr = np.array([r_mid, r_starr[1]])
        else:
            r_starr = np.array([r_starr[0], r_mid])
            
        r_mid = np.mean(r_starr)
        bond1d = np.sum(c1b*ZCB(r_mid, rbar1, k1, v1, t1b, t1c).price())
        i += 1
        if i == 20:
            break
        
    r_star = r_mid
    
    
    ##8 strike prices
    kkk = c1b*ZCB(r_star, rbar1, k1, v1, t1b, t1c).price()

    ##Reuse the rate paths from 1c
    pv = np.zeros((m,8))
    for i in range(0,8):
        bond = c1b[i]*ZCB(rmodel1c[:,-1],rbar1, k1, v1, t1b[i], t1c).price()
        pf = np.maximum(bond-kkk[i],0)
        pv[:,i] = pf*rrr
    
    call1d = np.mean(np.sum(pv, axis = 1))
    
    '''Summary'''
    print('Question 1: Vasicek Model')
    print('1a. The pure discount bond is worth ' + str(round(price1a,6)))
    print('1b. The Coupon Paying Bond is worth ' + str(round(price1b,6)))
    print('1c. The European call on pure discount bond is ' + str(round(call1c,6)))
    print('1d. The European call on coupon paying bond is ' + str(round(call1d,6)))    
    print('-------------------------------------------------------------------')
    
    '''Question 2: CIR Model'''
    r2 = 0.05
    v2 = 0.18
    k2 = 0.92
    rbar2 = 0.055
    f2 = 10000
    x2 = 9800 #strike price
    
    t2a = 0.5
    tt2a = 1
    n2a = int(t2a/dt)
    
    '''2(a). European Call on a Pure Discount Bond (Nested Simulation)'''    
    m2 = 1000 #1st-layer simulation
    mm2 = 1000 #2nd-layer simulation
    
    '''1st-layer Simulation'''
    ##Reuse the normal variable from Question 1b
    rmodel2a = rates(m2, r2, rbar2, v2, k2, z2, t2a, dt, method = 'cir', 
                     original = True).model()
    
    '''2nd-layer Simulation'''
    size = int(mm2*(tt2a-t2a)/dt)
    pbond2a = []
    for mm in range(0, m2):
        r2m = rmodel2a[mm,-1]
        np.random.seed(mm)
        zz2 = np.random.normal(size = size)
        rrmodel2a = rates(mm2, r2m, rbar2, v2, k2, zz2, tt2a-t2a, dt, 
                          method = 'cir', original = True).model()[:,1:]
        pbond2a.append(np.mean(f2*np.exp(-dt*np.sum(rrmodel2a, axis = 1))))

        
    payoff2a = (np.array(pbond2a) - x2)
    payoff2a[payoff2a < 0] = 0
    
    call2a = np.mean(np.exp(-dt*np.sum(rmodel2a[:,1:], axis = 1))*payoff2a)

    
    '''2(b). European Call on a Pure Discount Bond (Explicit Formula)'''
    pbond2b = f2*ZCB(rmodel2a[:,n2a],rbar2, k2, v2, tt2a-t2a).price()
        
    payoff2b = pbond2b - x2
    payoff2b[payoff2b < 0] = 0
    
    call2b = np.mean(np.exp(-dt*np.sum(rmodel2a[:,1:n2a+1], axis = 1))*payoff2b)
    
    '''Summery'''
    print('Question 2: CIR Model')
    print('The European call value of the Pure Discount Bond using simulations \
is ' + str(round(call2a, 6)))
    print('The European call value of the Pure Discount Bond using explicit \
formula is ' + str(round(call2b, 6)))
    print('-------------------------------------------------------------------')
    
    
    
    '''Question 3: G2++ Model and European Put'''
    x0 = 0
    y0 = 0
    phi = 0.03 #constant
    r0 = 0.03
    rho = 0.7
    a = 0.1
    b = 0.3
    v3 = 0.03
    eta = 0.08
    x3 = 985
    f3 = 1000
    
    t3 = 0.5
    tt3 = 1
    
    n3 = int(t3/dt)
    
    '''Method 1'''
    np.random.seed(7)
    z4 = np.random.normal(size = int(m*n3*2*2))
    rmodel4 = G2plusplus(m, x0, y0, r0, phi, rho, a, b, v3, eta, tt3, dt, z4)
    
    pbond4 = f3*np.exp(-dt*np.sum(rmodel4[:,n3+1:], axis = 1))
    
    payoff = x3 - pbond4
    payoff[payoff < 0] = 0
    
    put33 = np.mean(payoff*np.exp(-dt*np.sum(rmodel4[:,1:n3+1], axis = 1)))
    
    '''Method 2'''
    '''1st-layer Simulation'''
    ##Double simulation
    m3 = 1000
    mm3 = 1000
    
    
    np.random.seed(1357)
    z3 = np.random.normal(size = int(m3*n3*2))
    rmodel3 = G2plusplus(m3, x0, y0, r0, phi, rho, a, b, v3, eta, t3, dt, z3)
    xmodel, ymodel = G2plusplus(m3, x0, y0, r0, phi, rho, a, b, v3, eta, t3, 
                                dt, z3, outputr= False)
    
    '''2nd-layer Simulation'''
    size3 = int(mm3*(tt3-t3)/dt)
    pbond3 = [] 
    for mm in range(0, m3):
        xx0 = xmodel[mm,-1]
        yy0 = ymodel[mm,-1]
        rr0 = rmodel3[mm, -1]
        
        np.random.seed(mm)
        zz3 = np.random.normal(size = size3*2)
        rrmodel3 = G2plusplus(mm3, xx0, yy0, rr0, phi, rho, a, b, v3, eta, 
                              tt3-t3, dt, zz3)[:,1:]
        
        pbond3.append(np.mean(f3*np.exp(-dt*np.sum(rrmodel3, axis = 1))))
        
    payoff3 = x3 - np.array(pbond3)
    put3 = np.mean(np.maximum(payoff3,0)*np.exp(-dt*np.sum(rmodel3[:,1:], axis = 1)))
    
    print('Question 3: G2++ Model')
    print('The European Put of the Pure Discount Bond is ' + str(round(put3, 6)))
    print('------------------------------------------------------------------')
    
    