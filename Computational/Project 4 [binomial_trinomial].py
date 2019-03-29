# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:30:51 2019

@author: mzczm
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

'''2/19/2019 Update: Revise the functino geoBrownian()
change the output data from df to narray, which improves the running time by 90%
Please see the copy of the original function in Project 2.py'''

#-----------------------------New Function Area-------------------------------#

def binomialTree(s, k, rf, t, n, u, d, p, option = "call", style = "European", tree = False):
    '''binomial tree model to price non-dividend paying stock options'''
    
    para = [s, k, rf, t, n, u, d, p]
    
    for a in para:
        if a <= 0:
            print("Invalid input! Can't be less than 0")
            return None
        
    if type(n) != int:
        print("Parameter n is not an integer")
        n = int(n)
        
    dt = t/int(n)
    
    ##model would be the price of the underlying stock
    model = np.zeros((n+1,n+1))
    model[n][0] = s
    
    r = n
    while r >= 0:
        if r == n:
            for c in range(1, n+1):          
                model[r][c] = model[r][c-1]*d
        else:
            for c in range(n-r,n+1):
                model[r][c] = model[r+1][c-1]*u
        r -= 1   
    
    ##vmodel would be the value of the option
    vmodel = model.copy()
    
    if option == "call":
   
        for r in range(0, n+1):
            vmodel[r][n] = max(model[r][n]-k, 0)
            
    elif option == "put":
        
        for r in range(0, n+1):
            vmodel[r][n] = max(k-model[r][n], 0)        
        
    else:
        print("Invalid option input, must be either call or put")
        return None
    
    if style == "European":
        c = n-1
        while c >= 0:
            for r in range(n-c, n+1):
                vmodel[r][c] = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p+vmodel[r][c+1]*(1-p))
            c -= 1  
    elif style == "American":
        c = n-1
        if option == "call":
            while c >= 0:
                for r in range(n-c, n+1):
                    ex = max(model[r][c]-k, 0)
                    no = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p+vmodel[r][c+1]*(1-p))
                    vmodel[r][c] = max(ex, no) 
                c -= 1
        elif option == "put":
            while c >= 0:
                for r in range(n-c, n+1):
                    ex = max(k-model[r][c], 0)
                    no = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p+vmodel[r][c+1]*(1-p))
                    vmodel[r][c] = max(ex, no) 
                c -= 1    
    else:
        print("Invalid style input, must be either European or American")
        return None
    
    if tree == True:
        return vmodel ##The entire tree
    else:
        return vmodel[n][0] ##Option price

def trinomialTree(s, k, rf, t, n, u, d, p, log = False, option = "call", style = "European", tree = False):
    '''Trinomial Tree to price option'''
    
    dt = t/int(n)   
    
    if len(p) != 3:
        print("p does not have enough value")
        return None
    
    model = np.zeros((n*2+1, n+1))
    
    model[n] = s
    
    if log == False:
        para = [s, k, rf, t, n, u, d]
        
        for a in para:
            if a <= 0:
                print("Invalid input! Can't be less than 0")
                return None        
        
        for c in range(1, n+1):
            model[n-c][c:] = model[n-c+1][c-1]*u
            model[n+c][c:] = model[n+c-1][c-1]*d
    elif log == True:
         for c in range(1, n+1):
            model[n-c][c:] = model[n-c+1][c-1]*np.exp(u)
            model[n+c][c:] = model[n+c-1][c-1]*np.exp(d)       
        
    vmodel = model.copy()
    
    if option == "call":
        for r in range(0, n*2+1):
            vmodel[r][n] = max(model[r][n] - k, 0)
    elif option == "put":
        for r in range(0, n*2+1):
            vmodel[r][n] = max(k - model[r][n], 0)
    else:
        print("Invalid option")
        return None
    
    if style == "European":
        c = n-1
        while c >= 0:
            for r in range(n-c,n+c+1):
                vmodel[r][c] = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p[0] + \
                                              vmodel[r][c+1]*p[1] + \
                                              vmodel[r+1][c+1]*p[2])
            c -= 1    
    elif style == "American":
        c = n-1
        if option == "call":
            while c >= 0:
                for r in range(n-c,n+c+1):
                    ex = max(model[r][c]-k, 0)
                    no = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p[0] + \
                                               vmodel[r][c+1]*p[1] + \
                                               vmodel[r+1][c+1]*p[2])
                    vmodel[r][c] = max(ex, no)
                c -= 1  
        elif option == "put":
            while c >= 0:
                for r in range(n-c,n+c+1):
                    ex = max(k-model[r][c], 0)
                    no = np.exp(-rf*dt)*(vmodel[r-1][c+1]*p[0] + \
                                               vmodel[r][c+1]*p[1] + \
                                               vmodel[r+1][c+1]*p[2])
                    vmodel[r][c] = max(ex, no)
                c -= 1             
    
    if tree == False:
        return vmodel[n][0]
    else:
        return vmodel

##---------------------------Function section ends---------------------------##
    
##---------------------------Old Functions-----------------------------------##

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

'''Revised geoBrownian() function'''
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


def halton(size, base):
    '''generate a halton sequence of size specified'''
    
    import numpy as np
    
    seq = np.zeros(size)
    numbits = int(1 + np.ceil(np.log(size)/np.log(base)))
    vetbase = 1/(base**np.arange(1,numbits+1))
    workvet = np.zeros(numbits)
    for i in range(0, size):
        j = 0
        ok = 0
        while ok == 0:
            workvet[j] += 1
            if workvet[j] < base:
                ok = 1
            else:
                workvet[j] = 0
                j += 1
        seq[i] = np.dot(workvet, vetbase)
    
    return seq

##----------------------------Old Function Ends------------------------------##

##----------------------------Script section begins-------------------------##

if __name__ == '__main__':
    
    
    '''Question 1: European Call Binomial Method'''
    r1 = 0.05
    t1 = 0.5 #6-month
    v1 = 0.24
    s1 = 32
    k1 = 30
    
    n1 = [10,20,40,80,100,200,500]
    
    ##(a) When u = 1/d
    output_1a = []
    for n in n1:
        dt = t1/n
        c = (np.exp(-r1*dt) + np.exp((r1+v1**2)*dt))/2
        
        d = c - np.sqrt(c**2-1)
        u = 1/d
        
        p = (np.exp(r1*dt)-d)/(u-d)
        
        output_1a.append(binomialTree(s1,k1,r1,t1,n,u,d,p))
        
    ##(b)
    output_1b = []
    for n in n1:
        dt = t1/n
        
        u = np.exp(r1*dt)*(1+np.sqrt(np.exp((v1**2)*dt)-1))
        d = np.exp(r1*dt)*(1-np.sqrt(np.exp((v1**2)*dt)-1))
        
        p = 0.5
        
        output_1b.append(binomialTree(s1,k1,r1,t1,n,u,d,p))    
        
    ##(c): JR Model
    output_1c = []
    for n in n1:
        dt = t1/n
        
        u = np.exp((r1-(v1**2)/2)*dt + v1*np.sqrt(dt))
        d = np.exp((r1-(v1**2)/2)*dt - v1*np.sqrt(dt))
        
        p = 0.5
        
        output_1c.append(binomialTree(s1,k1,r1,t1,n,u,d,p))        
    
    ##(d): CRR Model
    output_1d = []
    for n in n1:
        dt = t1/n
        
        u = np.exp(v1*np.sqrt(dt))
        d = np.exp(-v1*np.sqrt(dt))
        
        p = 1/2 + ((r1-(v1**2)/2)*np.sqrt(dt)/v1)/2
        
        output_1d.append(binomialTree(s1,k1,r1,t1,n,u,d,p))      
            
    ##Summary for problem 1
    summary_1 = pd.DataFrame({"Method 1": output_1a, "Method 2": output_1b, 
                              "Method 3": output_1c, "Method 4": output_1d}, 
                                index = n1)
    print(summary_1)
    
    plt.figure()
    plt.plot(summary_1)
    plt.xticks(n1, fontsize = "small")
    plt.title("European call price under 4 binomial methods")
    plt.legend(["Method 1","Method 2", "Method 3", "Method 4"])
    
    
    '''Question 2: GOOG'''
    goog = pd.read_csv("S:/UCLA/405 Computational Methods In Finance/Week 4/GOOG.csv")
    
    r2 = 0.02
    #On yahoo finance, the 2020 Jan option expires on 01/17/2020
    expiration = datetime.date(2020,1,17)
    now = datetime.date(2019,2,6)
    t2 = (np.busday_count(now, expiration) - 8)/252 #less 8 holidays
    goog['return'] = goog["Adj Close"]/goog["Adj Close"].shift(1)-1
    v2 = np.std(goog['return'])*np.sqrt(252) ##annualized volatility
    s2 = goog.iloc[-1,5] #closing price on 1/6/2019
    k2 = round(1.1*s2,-1) #1230
    
    n2 = 1000
    
    ##(a) Binomial Method
    ##For this question, I chose CRR Model
    dt2 = t2/n2
        
    u2 = np.exp(v2*np.sqrt(dt2))
    d2 = np.exp(-v2*np.sqrt(dt2))
        
    p2 = 1/2 + ((r2-(v2**2)/2)*np.sqrt(dt2)/v2)/2
    
    goog_2a = binomialTree(s2,k2,r2,t2,n2,u2,d2,p2)
    goog_2a_yahoo = 68 #last price
    
    ##(b) Find the implied volatility
    ##Given the fact the option market price is higher
    ##We expect the implied volatility is higher
    ##Through some testing, we can tell that the implied volaility is between
    ##[1*v, 1.02*v] where v is the current volaility
    
    trange = [1*v2, 1.02*v2]
    
    goog_t = goog_2a
    ctrl = 0
    target = float(goog_2a_yahoo)
    while round(goog_t,6) != target:
        ##Using the binary search algorithm to find the implied vol
        v = np.mean(trange)
        u2b = np.exp(v*np.sqrt(dt2))
        d2b = np.exp(-v*np.sqrt(dt2))
        
        p2b = 1/2 + ((r2-(v**2)/2)*np.sqrt(dt2)/v)/2
        goog_t = binomialTree(s2,k2,r2,t2,n2,u2b,d2b,p2b)
        
        if round(goog_t, 6) < target:
            trange[0] = v
        elif round(goog_t, 6) > target:
            trange[1] = v
        else:
            implied_vol_2b = v
            
        ctrl+=1
        if ctrl == 25: ##To prevent the program to run endlessly
            print("No find")
            break
        
    
    ##Summary for Question 2
    summary_2 = pd.DataFrame({"Call Price":[goog_2a, goog_2a_yahoo],
                              "(Implied) Vol": [v2, implied_vol_2b]}, 
                            index = ["Estimate", "Market Price"])
    print(summary_2)
            
    '''Question 3: Option and Greeks''' 
    s3 = 49
    k3 = 50
    r3 = 0.03
    v3 = 0.2
    t3 = 0.3846
    mu3 = 0.14
    epsilon = 0.005 #I set the epsilon as 0.5% of whatever values it is applying to
    sss = range(20, 81, 2)
    
    ##For this question, I chose to use JR model
    n3 = 100
    p3 = 0.5
    
    dt3 = t3/n3
    
    u3 = np.exp((r3-(v3**2)/2)*dt3 + v3*np.sqrt(dt3))
    d3 = np.exp((r3-(v3**2)/2)*dt3 - v3*np.sqrt(dt3))
    
    p3 = 0.5
    
    
    delta_s = []
    gamma3 = []
    
    for s in sss:
         
        ds = 0.1 ##When you increase the ds to around 1 to 1.5, the plot smooths out
        s_range = ds*2
        s_plus = s+ds
        s_minus = s-ds

        c_p = binomialTree(s_plus,k3,r3,t3,n3,u3,d3,p3)
        c_m = binomialTree(s,k3,r3,t3,n3,u3,d3,p3)
        c_d = binomialTree(s_minus,k3,r3,t3,n3,u3,d3,p3)
        ##Estimate delta as a function of s
        dd = (c_p - c_d)/s_range
        delta_s.append(dd)
        
        ##Estimate gamma
        g = ((c_p-c_m) - (c_m - c_d))/((s_range/2)**2)
        gamma3.append(g)

    
    ##Estimate delta as a function of t
    s_range = s3*epsilon*2
    s_plus = s3*(1+epsilon)
    s_minus = s3*(1-epsilon)
    
    delta_t = []    
    for t in np.arange(0, t3, 0.01):
        ddt3 = t/n3
        
        u3t = np.exp((r3-(v3**2)/2)*ddt3 + v3*np.sqrt(ddt3))
        d3t = np.exp((r3-(v3**2)/2)*ddt3 - v3*np.sqrt(ddt3))
        
        if t == 0:
            c_p = max(s_plus-k3,0)
            c_d = max(s_minus-k3,0)
        else:
            c_p = binomialTree(s_plus,k3,r3,t,n3,u3t,d3t,p3)
            c_d= binomialTree(s_minus,k3,r3,t,n3,u3t,d3t,p3)
        
        ddt = (c_p - c_d)/s_range
        delta_t.append(ddt)
        
    ##Estimate theta
    t3_plus = t3+1/365/2
    t3_minus = t3-1/365/2
    
    u3t_p = np.exp((r3-(v3**2)/2)*t3_plus/n3 + v3*np.sqrt(t3_plus/n3))
    d3t_p = np.exp((r3-(v3**2)/2)*t3_plus/n3 - v3*np.sqrt(t3_plus/n3))
        
    u3t_m = np.exp((r3-(v3**2)/2)*t3_minus/n3 + v3*np.sqrt(t3_minus/n3))
    d3t_m = np.exp((r3-(v3**2)/2)*t3_minus/n3 - v3*np.sqrt(t3_minus/n3))
    
    theta = []
    for s in sss:
        t = (binomialTree(s,k3,r3,t3_plus,n3,u3t_p,d3t_p,p3) - \
             binomialTree(s,k3,r3,t3_minus,n3,u3t_m,d3t_m,p3))/(-2*t3)
        theta.append(t)
        
    ##Estimate vega
    v_range = v3*epsilon*2
    v3_plus = v3*(1+epsilon)
    v3_minus = v3*(1-epsilon)
    
    u3v_p = np.exp((r3-(v3_plus**2)/2)*dt3 + v3_plus*np.sqrt(dt3))
    d3v_p = np.exp((r3-(v3_plus**2)/2)*dt3 - v3_plus*np.sqrt(dt3))
        
    u3v_m = np.exp((r3-(v3_minus**2)/2)*dt3 + v3_minus*np.sqrt(dt3))
    d3v_m = np.exp((r3-(v3_minus**2)/2)*dt3 - v3_minus*np.sqrt(dt3))
    
    vega = []
    for s in sss:  
        v = (binomialTree(s,k3,r3,t3,n3,u3v_p,d3v_p,p3) - \
             binomialTree(s,k3,r3,t3,n3,u3v_m,d3v_m,p3))/v_range
        vega.append(v/100)  
        
    ##Estimate rho
    r_range = r3*epsilon*2
    r3_plus = r3*(1+epsilon)
    r3_minus = r3*(1-epsilon)
    
    u3r_p = np.exp((r3_plus-(v3**2)/2)*dt3 + v3*np.sqrt(dt3))
    d3r_p = np.exp((r3_plus-(v3**2)/2)*dt3 - v3*np.sqrt(dt3))
        
    u3r_m = np.exp((r3_minus-(v3**2)/2)*dt3 + v3*np.sqrt(dt3))
    d3r_m = np.exp((r3_minus-(v3**2)/2)*dt3 - v3*np.sqrt(dt3))
    
    rho = []
    for s in sss:
        rr = (binomialTree(s,k3,r3_plus,t3,n3,u3r_p,d3r_p,p3) - \
             binomialTree(s,k3,r3_minus,t3,n3,u3r_m,d3r_m,p3))/r_range
        rho.append(rr/100)
        
        
    ##Summary
    summary_3 = pd.DataFrame({"Delta_d(s)": delta_s, "Theta": theta, 
                              "Gamma": gamma3,"Vega": vega, "Rho": rho}, index = sss)
    
    plt.figure()
    f, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5,1, sharex = True)
    ax1.plot(summary_3['Delta_d(s)'])
    ax1.set_title('Option Greeks')
    ax1.legend('Delta', loc = 4)
    ax2.plot(summary_3['Gamma'], c= 'r')
    ax2.legend('Gamma')
    ax3.plot(summary_3['Theta'], c= 'g')
    ax3.legend('Theta', loc = 3)
    ax4.plot(summary_3['Vega'], c= 'yellow')
    ax4.legend('Vega')
    ax5.plot(summary_3['Rho'], c = 'grey') 
    ax5.legend('Rho', loc = 4)
    ax5.set_xlabel('S')
    
    
    summary_3ii = pd.DataFrame({'Delta_d(t)': delta_t}, index = np.arange(0,t3, 0.01))
    plt.figure()
    plt.plot(summary_3ii)
    plt.title("delta as a function of t")
    plt.xlabel("t")
    
        
        
    '''Question 4: Put Option'''
    r4 = 0.05
    v4 = 0.3
    k4 = 100
    s4 = range(80,120+1,4)
    t4 = 1 #12 months
    n4 = 1000
    dt4 = t4/n4
    
    ##For this question, I am using the JR Model    
    u4 = np.exp((r4-(v4**2)/2)*dt4 + v4*np.sqrt(dt4))
    d4 = np.exp((r1-(v4**2)/2)*dt4 - v4*np.sqrt(dt4))
    p4 = 0.5
    
    ##European put option
    output_4euro = [binomialTree(s,k4,r4,t4,n4,u4,d4,p4, option = "put") for s in s4]
    
    ##American put option
    output_4amer = [binomialTree(s,k4,r4,t4,n4,u4,d4,p4, option = "put", \
                                 style = "American") for s in s4]
    
    ##Summary
    summary_4 = pd.DataFrame({"European Put": output_4euro, 
                              "American Put": output_4amer}, index = s4)
    print(summary_4)
    
    plt.figure()
    plt.plot(summary_4)
    plt.title("Put Option Price: European Vs. American")
    plt.xticks(s4)
    plt.legend(["European", "American"])
    
    
    
    '''Question 5: Trinomial Model'''
    r5 = 0.05
    v5 = 0.24
    s5 = 32
    k5 = 30
    t5 = 0.5 #6-month
    n5 = [10, 15, 20, 40, 70, 80, 100, 200, 500]
    
    ##(a)
    output_5a = []
    for n in n5:
        dt5 = t5/n
        
        d = np.exp(-v5*np.sqrt(3*dt5))
        u = 1/d
        
        p_u = (r5*dt5*(1-d) + (r5*dt5)**2 + (v5**2)*dt5)/((u-d)*(u-1))
        p_d = (r5*dt5*(1-u) + (r5*dt5)**2 + (v5**2)*dt5)/((u-d)*(1-d))
        p_m = 1 - p_u - p_d
        p = [p_u, p_m, p_d]
    
        output_5a.append(trinomialTree(s5, k5, r5, t5, n, u, d, p))
    
    ##(b)
    output_5b = []
    for n in n5:
        dt5 = t5/n
        
        xu = v5*np.sqrt(3*dt5)
        xd = -v5*np.sqrt(3*dt5)
        
        gamma = r5-(v5**2)/2
        p_u = 0.5*(((v5**2)*dt5+(gamma**2)*(dt5**2))/(xu**2) + (gamma*dt5)/xu)
        p_d = 0.5*(((v5**2)*dt5+(gamma**2)*(dt5**2))/(xu**2) - (gamma*dt5)/xu)
        p_m = 1 - p_u - p_d
        p = [p_u, p_m, p_d]
        
        output_5b.append(trinomialTree(s5, k5, r5, t5, n, xu, xd, p, log = True))
    
    ##Summary
    summary_5 = pd.DataFrame({"Method 1": output_5a, "Method 2": output_5b},
                             index = n5)
    print(summary_5)
    
    plt.figure()
    plt.plot(summary_5)
    plt.title("Trinomial Tree Model")
    plt.xticks(n5, fontsize = "x-small")
    plt.legend(["Method 1", "Method 2"])
    
    
    
    '''Question 6: Halton and European Call Price'''
    seed1 = 2
    seed2 = 7
    s6 = 66
    k6 = 77
    t6 = 1
    r6 = 0.02
    sigma6 = 0.25
    n6 = 100 #100 intervals
    m6 = 2000 #simulate 1000 paths
    size = int(n6*m6)
    
    h1 = halton(int(size/2),seed1)
    h2 = halton(int(size/2),seed2)
    
    ##Because my normal variables funnction normal() takes on 
    ##uniform[i] and uniform[i+1] when generating random variables
    ##So in order to make sure H1 and H2 is used in the right place
    ##I used the following function to alternate h1 and h2
    h_all = [h1[int(i/2)] if i%2 == 0 else h2[int(i/2)] for i in range(0, size)]
    
    z6 = normal(size = size, uniform = h_all)
    
    np.random.seed(17)
    ##Because z6 is generated using quasi-random variables
    ##When I try to simulate s_t, I need to break the ordres of z6 again 
    ##So to make the random variables in z6 does not follow a certain order pattern
    s_t = geoBrownian(n = m6, start = s6, r = r6, sigma = sigma6, t = t6, 
                normal = np.random.choice(z6, size = size, replace = False), 
                divide = n6, single = True)
    payoff = [max(s-k6, 0) for s in s_t]
    
    c_q6 = np.exp(-r6*t6)*np.mean(payoff)
    
    print("Question 6: The call price is " + str(round(c_q6,6)))
    

