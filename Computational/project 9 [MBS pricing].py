# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:17:42 2019

@author: Serena Peng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#-------------------------------New Function----------------------------------#

def numerix(pv0, wac, t, dt, rmodel, m = 1, r10 = None):
    '''Use numerix prepayment model to calculate mortgage price
    m is the # of simulation
    r10 must be a m x n matrix (np.darray)'''
    
    try:
        n = int(t/dt)
    except:
        print('Invalid t or dt input')
        return None
    
    if (n <= 1) or (wac < 0) or (pv0 <= 0):
        print('Invalid negative input wac or pv0')
        return None
    
    try:
        if (rmodel.shape[0]) == m and (rmodel.shape[1] < n):
            print('# of rmodel are not sufficient')
            return None
    except:
        print('rmodel must be of type np.darray')
        return None
    
    
    if type(r10) == type(None):
        r10 = np.zeros((m, n))
        for c in range(1, n+1):
            r10[:,c-1] = np.mean(rmodel[:,c-1:c-1+120],axis = 1)
#        r10 = rmodel[:,:360].copy()
    
    try:
        if r10.shape != (m, n):
            print('# of r10 are not sufficient')
            return None
    except:
        print('rates r10 must be of type np.darray')
        return None
    
    pvmodel = np.zeros((m, n+1))
    pvmodel[:,0] = pv0
    
    #Numerix Prepayment Model variables:
    ri = 0.28 + 0.14 * np.arctan(-8.57 + 430 * (wac - r10))
#    del r10
    sg = np.minimum(np.arange(1, n+1)/30,1)
    sy = np.tile(np.array([0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 
                    1.18, 1.22, 1.23, 0.98]), int(t))
    sy = sy[:n]
 
    cashflow = np.zeros((m, n))
    r_discount = np.zeros((m, n))
    for c in range(1, n+1):
        pmt = (pvmodel[:,c-1]*wac/12)/(1-(1/((1+wac/12)**(n-c+1)))) #payment
        bu = 0.3 + 0.7 * (pvmodel[:,c-1]/pv0) #burnout
        cpr = ri[:,c-1]*bu*sg[c-1]*sy[c-1] #conditional prepayment rate
        ip = wac/12*pvmodel[:,c-1] #interest payment
        sp = pmt - ip #schedule princpal
        pp = (pvmodel[:, c-1] - sp)*(1-(1-cpr)**(1/12)) #prepayment
        pvmodel[:, c] = np.maximum(pvmodel[:, c-1] - (sp + pp),0) #track loan principal
        r_discount[:,c-1] = np.exp(-dt*np.sum(rmodel[:,1:c+1], axis = 1)) #discount rate for cashflow
        cashflow[:, c-1] = np.maximum(ip, 0) #cash flow c = TPP + IP
    
    cashflow += -np.diff(pvmodel) #cash flow c = TPP + IP
    
    return np.mean(np.sum(r_discount*cashflow, axis = 1))    

#-------------------------------Old Function---------------------------------#
def uniform(size, seed, a = 7**5, b = 0, m = 2**23-1):
    '''return a random variables u~u[0,1]
    The defaul a, b, and b is using LGM method parameters'''
    
    if (seed == 0) and (b==0):
        print("Input error!")
        return None
    
    if size <= 0:
        print("Invalid size")
        return None
    
    x_n = [seed]
    
    for i in range(1, size+1):
        x_n.append((a*x_n[i-1]+b)%m)
        
    return np.array(x_n[1:])/m

def normal(size, uniform, method = "BM", loop = False):
    '''Two possible methods: default is Box-Muller
    BM stands for Box-Muller method; PM stands for Polar-Marsaglia method.
    The loop arg is only for PM method'''
    
    try:
        if size > len(uniform):
            print("Insufficient uniform random variables")
            return None
        elif size <= 0:
            print("Invalid size")
            return None
    except:
        print("Input error!")
        return None
    
    import math
    if method == "BM":
        pi = math.pi
        
        output = []
        i = 0
        while i < size:
            u1 = uniform[i]
            try:
                u2 = uniform[i+1]
            except:
                #When you generate odd number of variables, 
                #the last uniform variable will be first variable
                u2 = uniform[0]

            #Calculate Z1
            output.append(np.sqrt(-2*math.log(u1))*math.cos(2*pi*u2))
            #Calculate Z2
            output.append(np.sqrt(-2*math.log(u1))*math.sin(2*pi*u2))
            i += 2
            
        return output[:size]
    elif method == "PM":
        output = []
        i = 0
        while len(output) < size:
            try:
                v1 = 2*uniform[i]-1
            except:
                if loop == True:
                    #loop again; start from index 1 (the 2nd variable)
                    i = 1
                    v1 = 2*uniform[i]-1
                else:
                    break
            
            try:
                v2 = 2*uniform[i+1]-1
            except:
                #When the size of uniform is odd, 
                #the last uniform variable will be first variable
                v2 = 2*uniform[0]-1
                if loop == True:
                    #when loop = True, it means that the function will
                    #recycle the uniform variables until it generate 
                    #enough variables; So i will start from -1 because it will
                    #be added 2 shortly: -1+2 = 1. So the new loop will start
                    #from index 1 (the 2nd variable)
                    i = -1
            w = v1**2 + v2**2
            if w <= 1:
                #Calculate Z1
                output.append(v1*np.sqrt(-2*math.log(w)/w))
                #Calculate Z2
                output.append(v2*np.sqrt(-2*math.log(w)/w))
#            if i >= len(uniform)-2:
#                break
            i += 2  
                        
        return output[:size]
    else:
        print("Invalid method")
        return None



class rates:
    '''Simulate the rate paths using Vasicek or CIR model'''
    
    def __init__(self, m, r0, rbar, sigma, k, z, t, dt, method = 'vasicek', original = False):
        
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
            return np.mean(np.exp(-self.dt*np.sum(rmodel[:,1:], axis = 1)))
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
        elif self.method == 'cir':
            return self.__cir()
        else:
            print('Invalid method argument')
            return None
    
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
      
#---------------------------Script Zone---------------------------------------#

if __name__ == '__main__':
    
#    begin = time.time()
    
    wac = 0.08 #monthly cash flow
    r_m = wac/12
    pv0 = 100000
    
    r0 = 0.078
    k = 0.6
    rbar = 0.08
    v = 0.12
    t = 30 #30 years mortgage
    t10 = 10 #10-year treasury
#    t = tm + t10 #the total period needs to be simulated
    dt = 1/12 #monthly
    
    n_r = int(t/dt)#+120
    m = 5000
    
    n360 = int(t/dt) #360
    n10 = int(t10/dt) #10-year treasury
    
    size = int(m*n_r)
#    np.random.seed(7)
#    z = np.array(np.random.normal(size = int(m*n_r)))
    z = np.array(normal(size = size, uniform = uniform(size = size, seed = 111)))

    
    '''Question 1: Numerix Prepayment Model'''
    ##1b. With respect to k
    kr = np.arange(0.3, 0.905, 0.1)
    
    price1b = []
    for k1b in kr:
        rmodel = rates(m = m, r0 = r0, rbar = rbar, sigma = v, k = k1b, z = z, 
                       t = t, dt = dt, method = 'cir', original = True).model()
        r10 = ZCB(r = rmodel[:,1:], rbar = rbar, k = k1b, sigma = v, T = t10, 
                  method = 'cir').price()
        r10 = -1/10*np.log(r10)
            
        price1b.append(numerix(pv0 = pv0, wac = wac, t = t, dt = dt, 
                                 rmodel = rmodel, m = m,r10 = r10))
        
    summary1b = pd.DataFrame({'k': kr, 'Price': price1b})
    
    plt.figure()
    plt.plot(kr, price1b)
    plt.xlabel('k')
    plt.ticklabel_format(useOffset=False)
    plt.title('Mortgage price with respect to k = [0.3, 0.9]')
    
    
    ##1c. With respect to rbar
    rbarr = np.arange(0.03, 0.0905, 0.01)
    
    price1c = []
    for rbar1c in rbarr:
        rmodel = rates(m = m, r0 = r0, rbar = rbar1c, sigma = v, k = k, z = z, 
                       t = t, dt = dt, method = 'cir', original = True).model()
        r10 = ZCB(r = rmodel[:,1:], rbar = rbar1c, k = k, sigma = v, T = t10, 
                  method = 'cir').price()
        r10 = -1/10*np.log(r10)
            
        price1c.append(numerix(pv0 = pv0, wac = wac, t = t, dt = dt, 
                                 rmodel = rmodel, m = m, r10 = r10))
        
    summary1c = pd.DataFrame({'rbar': rbarr, 'Price': price1c})
    
    plt.figure()
    plt.plot(rbarr, price1c)
    plt.title('Mortgage price with respect to rbar = [0.03, 0.09]')
    plt.xlabel('r_bar')
    
    ##1d. With respect to sigma
    vr = np.arange(0.1, 0.205, 0.01)
    
    price1d = []
    for v1d in vr:
        rmodel = rates(m = m, r0 = r0, rbar = rbar, sigma = v1d, k = k, z = z, 
                       t = t, dt = dt, method = 'cir', original = True).model()
        r10 = ZCB(r = rmodel[:,1:], rbar = rbar, k = k, sigma = v1d, T = t10, 
                  method = 'cir').price()
        r10 = -1/10*np.log(r10)
            
        price1d.append(numerix(pv0 = pv0, wac = wac, t = t, dt = dt, 
                                 rmodel = rmodel, m = m, r10 = r10))
        
    summary1d = pd.DataFrame({'sigma': vr, 'Price': price1d})
    
    plt.figure()
    plt.plot(vr, np.array(price1d))
    plt.ticklabel_format(useOffset=False)
    plt.title('Mortgage price with respect to σ = [0.1, 0.2]')
    plt.xlabel('σ')
    
    print('Question 1:')
    print('The MBS price is {}'.format(round(summary1b['Price'][3], 3)))
    print('---------------------------------------------------------')
    
    del rmodel
    del r10
    
    '''Question 2. Option-Adjusted Spread (OAS)'''
    mbs = 110000.0
    
    ##current price is 101154.84.     
    rmodel2 = rates(m = m, r0 = r0, rbar = rbar, sigma = v, k = k, z = z, 
                   t = t, dt = dt, method = 'cir', original = True).model()
    r10_2 = ZCB(r = rmodel2[:,1:], rbar = rbar, k = k, sigma = v, T = t10, 
              method = 'cir').price()
    r10_2 = -1/10*np.log(r10_2)
        
    price2 = numerix(pv0 = pv0, wac = wac, t = t, dt = dt, rmodel = rmodel2, 
                       m = m, r10 = r10_2)
    
    ##find the OAS
    r_up = -0.009
    r_down = -0.017
    
    i = 0
    price2test = price2
    while round(price2test,6) != mbs:
        oas = np.mean([r_up, r_down])
        
        r10_oas = ZCB(r = rmodel2[:,1:] + oas, rbar = rbar, k = k, sigma = v, 
                      T = t10, method = 'cir').price()
        r10_oas = -1/10*np.log(r10_oas)
        
        price2test = numerix(pv0 = pv0, wac = wac, t = t, dt = dt, 
                             rmodel = rmodel2 + oas, m = m, r10 = r10_oas)
        
        if price2test < mbs:
            r_up = oas
        elif price2test > mbs:
            r_down = oas
            
        i +=1
        if i == 50: #prevent the loop to run infinitely
            break
    
    print('Question 2:')
    print('The OAS is {}%'.format(round(oas*100, 4)))
    print('-------------------------------------------------')
    
    del r10_2
    del r10_oas
    
    '''Question 3: Duration and Convexity'''
    y = 0.01/100*5 #5bp
    
    #reuse the rates model from question 2
    #price_up
    r10_up = ZCB(r = rmodel2[:,1:] + oas + y, rbar = rbar, k = k, sigma = v, 
                      T = t10, method = 'cir').price()
    r10_up = -1/10*np.log(r10_up)

    p_up = numerix(pv0 = pv0, wac = wac, t = t, dt = dt, rmodel = rmodel2 + oas + y, 
                   m = m, r10 = r10_up)
    
    #price_down
    r10_dn = ZCB(r = rmodel2[:,1:] + oas - y, rbar = rbar, k = k, sigma = v, 
                      T = t10, method = 'cir').price()
    r10_dn = -1/10*np.log(r10_dn)

    p_dn = numerix(pv0 = pv0, wac = wac, t = t, dt = dt, rmodel = rmodel2 + oas - y, 
                   m = m, r10 = r10_dn)
    
    #calculate
    duration = (p_dn - p_up)/(2*y*mbs)
            
    convexity = (p_dn + p_up - 2*mbs)/(2*(y**2)*mbs)
        
    print('Question 3:')
    print('The OAS duration is {}'.format(round(duration, 3)))
    print('The OAS convexity is {}'.format(round(convexity, 3)))
    
#    end = time.time()
    
