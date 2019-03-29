# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:34:24 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
#import sympy as sp

'''This file is comprised of three sections, new function, old function (from
previous project), and script section;
You can run the whole file at once. The expected runtime is about 9-12 minutes'''

#------------------------New Function Section------------------------------------#

class lsmc:
    
    def __init__(self,df, m, n, r, k, t, term = 3, option = "call", 
                 style = "American", method = "Monomials"):
        '''LSMC method for option pricing: 
        df = array of price paths
        m = # of paths
        n = # of time interval per path
        r = risk-free rate
        k = strike price
        t = maturity
        term = # of terms used in regression'''
        self.df = df
        self.m = m
        self.n = n
        self.r = r
        self.k = k
        self.t = t
        self.term = term
        self.option = option
        self.style = style
        self.method = method
    
    def __monomials__(self, iv):
        '''iv = independend variables, x'''
        ll = len(iv)
        
        return np.repeat(iv, self.term-1).reshape((ll, self.term-1))**np.arange(1,self.term)
    
    def __hermite__(self, iv):
        ll = len(iv)
        xx = np.repeat(iv, self.term).reshape((ll, self.term))
        
        xx[:,0] = 1
        xx[:,1] = 2*xx[:,1]
        for i in range(2, self.term):
            xx[:,i] = 2*iv*xx[:,i-1] - 2*(i+1-2)*xx[:,i-2]

        return xx[:,1:]
    
    def __laguerre__(self, iv):
        ll = len(iv)
        xx = np.repeat(iv, self.term).reshape((ll, self.term))        
#        x = sp.Symbol('x')
        i = 1
        while i <= self.term:
            if i == 1:
                xx[:,0] = np.exp(-iv/2)
            elif i == 2:
                xx[:,1] = np.exp(-iv/2)*(1-iv)
            elif i == 3:
                xx[:,2] = np.exp(-iv/2)*(1-2*iv+(iv**2)/2)
            elif i == 4:
                xx[:,3] = np.exp(-iv/2)*(1-3*iv+3*(iv**2)/2 - (iv**3)/6)
            elif i == 5:
                xx[:,4] = np.exp(-iv/2)*(1-4*iv+3*(iv**2) - 2*(iv**3)/3 + (iv**4)/24)
            else:
#                eq = sp.diff((x**i)*np.e**(-x), x, i-1)*(np.e**(x/2))
#                f = sp.lambdify(x, eq, 'numpy')
#                xx[:,i] = f(iv)/np.product(range(1,i))
                break
            i +=1
        
        return xx
    
    def model(self):

        if type(self.df) != np.ndarray:
            try:
                self.df = np.array(self.df)
            except:
                print("Input must be array")
                return None
        
        if (self.df.shape[0] != self.m) or (self.df.shape[1] != self.n+1):
            print("Insufficient df, given m and n")
            return None
        
        if type(self.term) != int:
            try:
                self.term = int(self.term)
                if self.term <=0:
                    print("term argument cannot be negative")
                    return None
            except:
                print("Invalid term argument")
                return None
        
        if self.method == "Laguerre":
            mm = False
        else:
            mm = True
            
        from sklearn.linear_model import LinearRegression
#        import sympy as sp
        
        dt = self.t/self.n
        
        cf = self.df.copy()
        
        if self.option == "call":
            cf = cf - self.k
        elif self.option == "put":
            cf = self.k - cf
        else:
            print("Invalid option input")
            return None
        
        cf[cf < 0] = 0 ##cf is the exercise value
        
        if self.style == "European":
            return np.mean(cf[:,self.n])*np.exp(-self.r*self.t)
        elif self.style == "American":
            c = self.n-1
            while c > 0:
                x = self.df[cf[:,c]>0, c]
                if len(x) == 0:
                    c-= 1
                    continue
                
                if self.method == "Monomials":
                    x = self.__monomials__(iv = x)
                elif self.method == "Hermite":
                    x = self.__hermite__(iv = x)
                elif self.method == "Laguerre":
                    x = self.__laguerre__(iv = x)
                else:
                    print("Invalid method argument")
                    return None
                pp = np.amax(cf[cf[:,c]>0,c+1:], axis = 1)
                try:
                    i = np.where(pp != 0)[1]
                except:
                    i = np.array([1]*len(pp))
                y = np.exp(-self.r*dt*i)*pp                
                linreg = LinearRegression(fit_intercept = mm).fit(X = x,y = y)
                
                no = linreg.intercept_ + np.sum(linreg.coef_*x, axis = 1)
                ex = cf[cf[:,c]>0,c]
                ex[ex < no] = 0
                cf[cf[:,c]>0, c] = ex
                cf[cf[:,c]>0, c+1:] = 0
                
                c-=1            
                    
            return sum(np.exp(-self.r*dt*np.arange(1,self.n+1))*np.mean(cf, axis = 0)[1:])


#-----------------------Old Functions Section--------------------------------#
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
        
    return [i/m for i in x_n[1:]]

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

def geoBrownian(n, start, r, sigma, t, normal, divide = 1, single = False):
    '''Simulate Geometric Brownian Motion process S_t
    n refers to the 
    When divide = 1, it will generate n S_t;
    When divide !=1, it will generate n paths of S_t
    When single == True, only return the last values for each path generated'''
    
    if (n <= 0) or (type(n) != int):
        print("Invalid size input")
        return None
    
    if (sigma < 0) or (start < 0) or (t < 0):
        print("Invalid negative input")
        return None
        
    if len(normal) < n*divide:
        print("Insufficient normal variables")
        return None
    
    if (type(divide) != int) & (type(divide) != float):
        print("Invalid divide input")
        return None
    
    import numpy as np
    normal = np.array(normal)
    
    if divide == 1:
            
        return start*np.e**((r-sigma**2/2)*t+sigma*np.sqrt(t)*normal[:n])
    
    elif divide > 1:
        output = np.zeros((n, int(divide+1)))
        output += start
        tint = t/divide
        trange = np.arange(tint, t+tint, tint)
        normal = normal[:int(divide*n)].reshape((n, divide))
        w = normal*np.sqrt(tint)
        del normal ##free up memory
        for d in range(1, int(divide)):
            w[:,d] += w[:, int(d-1)]
        ft = np.exp((r-(sigma**2)/2)*trange+sigma*w)
        output[:,1:] = output[:,1:]*ft
        
                
        if single == False:
            return output
        else:
            return output[:,-1]
    else:       
        print("Divde cannot be less than 1")
        return None
    

#------------------------Script Section starts--------------------------------#
if __name__ == '__main__':
    
    import time
    begin = time.time() ##Running time check
    
    '''Question 1: LSMC'''
    s1 = [36,40,44]
    k1 = 40
    r1 = 0.06
    v1 = 0.2
    dt = 0.01 #set dt as 0.01 by default
    kkk = [2,3,4] #How many terms used in the regression
    
    m1 = 100000
    t1 = [0.5, 1, 2]
    
    size = int(m1*t1[2]/dt/2)
    seed = 7
    ##Takes about 30 seconds to generate the following random variables
    z1 = np.array(normal(size = size, uniform = uniform(size = size, seed = seed)))
    
    methods = ['Laguerre', 'Hermite', 'Monomials']
    
    ##Use this nested loop to compute (a), (b), (c)

    output_1 = {}
    for s in s1:
        paths1 = geoBrownian(n = m1, start = s, r = r1, sigma = v1, t = t1[2],
                             normal = np.append(z1, -z1), divide = int(t1[2]/dt))
        for t in t1:
            n = int(t/dt)
            for kk in kkk:
                for mm in methods:
                    price_1 = lsmc(df = paths1[:,:n+1], m = m1, n = n, r = r1, 
                                   k = k1, term = kk, t = t, option = 'put', 
                                   method = mm).model()
                    
                    output_1["{}_s{}_t{}_k{}".format(mm, s, t, kk)] = price_1
    

    ##Summary
    laguerre = {}
    hermite = {}
    monomials = {}
    for i, j in output_1.items():
        if i.startswith('L'):
            laguerre[i] = j
        elif i.startswith('H'):
            hermite[i] = j
        else:
            monomials[i] = j
    
    index1 = np.array([idx[13:].split('_') for idx in laguerre.keys()][:9]).T.tolist()
    
    summary_1a = pd.DataFrame(np.array(list(laguerre.values())).reshape((3,9)).T).set_index(index1, drop = True)
    summary_1a.columns = ["S0 = {}".format(s) for s in s1]
    
    summary_1b = pd.DataFrame(np.array(list(hermite.values())).reshape((3,9)).T).set_index(index1, drop = True)
    summary_1b.columns = ["S0 = {}".format(s) for s in s1]
    
    summary_1c = pd.DataFrame(np.array(list(monomials.values())).reshape((3,9)).T).set_index(index1, drop = True)
    summary_1c.columns = ["S0 = {}".format(s) for s in s1]    
    
    print("Question 1:")
    print("1a. Laguerre Polynomials")
    print(summary_1a)
    print("----------------------------------------")
    print("1b. Hermite Polynomials")
    print(summary_1b)
    print("----------------------------------------")    
    print("1c. Simple Monomials")
    print(summary_1c)
    print("----------------------------------------")

    del paths1 #Free up memory
    
    '''Question 2: Forward-Start Option'''
    s2 = 65
    k2 = 60 #irrelevant for this question
    v2 = 0.2
    r2 = 0.06
    t2s = 0.2
    t2e = 1
    
    m2 = 10000
    dt = 0.01
    
    ##Reuse the z from question 1
    k_t = int(t2s/dt+1) #When the strike price is determined
    zz = int(m2*t2e/dt/2)
    paths2 = geoBrownian(n = m2, start = s2, r = r2, sigma = v2, t = t2e, 
                         normal = np.append(z1[:zz], -z1[:zz]), divide = int(t2e/dt))
    
    '''2a. European put option'''
    k2a = paths2[:,k_t]
    st2a = paths2[:,int(t2e/dt)]
    
    payoff = k2a - st2a
    payoff[payoff < 0] = 0
    
    put_2a = np.exp(-t2e*r2)*np.mean(payoff)
    
    '''2b. American put option'''
    t2b = int(t2e/dt-k_t)
    paths2b = paths2[:,k_t:] - np.repeat(k2a, repeats = int(t2b+1)).reshape((m2,int(t2b+1)))
    
    put_2b = np.exp(-r2*t2s)*lsmc(df = paths2b, m = m2, n = int(t2e/dt-k_t), 
                    r = r2, k = 0, t = t2e-t2s-dt, option = "put").model()
    
    ##Summary
    print("Question 2:")
    print("The value of forward-start European put is " + str(round(put_2a, 6)))
    print("The value of forward-start American put is " + str(round(put_2b, 6)))  
    
    end = time.time()
    
    