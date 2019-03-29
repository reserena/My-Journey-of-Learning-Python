# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:51:05 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Instruction: In this project, there are three sections
##First section: Reused functions from previous projects;
##Second section: New functions created for this project;
##Last section: Script;
'''You can run the whole file at once'''

'''2/19/2019 Update: Revise the functino geoBrownian()
change the output data from df to narray, which improves the running time by 90%
Please see the copy of the original function in Project 2.py'''

#----------------------Section 1: Old Functions------------------------------#

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
    '''Two possible methods:
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

'''Revised function geoBrownian()'''
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
        normal = normal[:int(divide*n)].reshape((n, int(divide)))
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
    
    
#-----------------------Section 2: New Functions-----------------------------#

def normal_cdf(x):
    '''Numerical Computation of N(.)'''
    d1 = 0.0498673470
    d2 = 0.0211410061
    d3 = 0.0032776263
    d4 = 0.0000380036
    d5 = 0.0000488906
    d6 = 0.0000053830
    if x >= 0:
        return 1-(1/2)*((1+d1*x+d2*(x**2)+d3*(x**3)+d4*(x**4)+d5*(x**5)+d6*(x**6))**(-16))
    else:
        return 1-normal_cdf(-x)
    
        
def bsm(s, k, r, t, sigma, d = 0, option = "call"):
    '''Using BSM model to calculate option price
    d is the continuous dividend yield'''
    
    if (s < 0) or (k < 0) or (r < 0) or (t <= 0) or (sigma <= 0) or (d < 0):
        print("Invalid negative input")
        return None
    
    import scipy.stats as st
    
    d1 = (np.log(s/k)+(r-d+sigma**2/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    if option == "call":
        return st.norm.cdf(d1)*s*np.exp(-d*t) - st.norm.cdf(d2)*np.exp(-r*t)*k
    elif option == "put":
        return st.norm.cdf(-d2)*np.exp(-r*t)*k - st.norm.cdf(-d1)*s*np.exp(-d*t)
    else:
        print("Invalid option input")
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
    
    
#---------------------------Script Zone--------------------------------------#

if __name__ == '__main__':
    
    ##Create normally distributed variables for future uses
    n = 1000000
    seed1 = 7
    seed2 = 17
    z = normal(size = n, uniform = uniform(size = n, seed = seed1))
    w = normal(size = n, uniform = uniform(size = n, seed = seed2))
    
    d = 100 ##Set the dt interval to be 100 per 1 t
    dt = 1/d
    dt_s = np.sqrt(dt)
    
    m1 = 1000 #numbers of simulation
    
    '''Question 1: E[x] and Probabilities'''
    '''Using Euler's Scheme'''
    x0 = 1
    y0 = 3/4
    
    ##Simulate Y2 and Y3
    t_y2 = 2
    t_y3 = 3
    y2 = []
    y3 = []
    j = 0
    for m in range(0, m1):
        y_t = y0
        for i in range(0, d*t_y3):
            t = i*dt
            if t == 2:
                y2.append(y_t)
            y_t = y_t + ((2/(1+t))*y_t+(1+t**3)/3)*dt + (1+t**3)/3*(dt_s*z[j])
            j += 1
             
        y3.append(y_t)
        
    Prob = len([1 for y in y2 if y > 5])/m1
    E2 = np.mean(y3) ##For E[Y3]
    
    ##Simulate X2
    t_x2 = 2
    x2 = []
    j = 0
    for m in range(0, m1):
        x_t = x0
        for i in range(0, d*t_x2):
            x_t = x_t + (1/5 - (1/2)*x_t)*dt + (2/3)*dt_s*w[j]
            j += 1
        x2.append(x_t)
        
    ##For some reason, Python throws warning when calculating x*(1/3) for a 
    ##a negative x. Therefore, for negative x, I calculate abs(x)**(1/3) first,
    ##and then convert to negative
    E1 = np.mean([x**(1/3) if x > 0 else -(-x)**(1/3) for x in x2])
    E3 = np.mean([x2[i]*y2[i] if x2[i] > 1 else 0 for i in range(0, m1)])
    
    ##Summery for Question 1
    print("For Question 1:")
    print("Prob = " + str(round(Prob,6))) 
    print("E1 = " + str(round(E1,6)))
    print("E2 = " + str(round(E2,6)))
    print("E3 = " + str(round(E3,6)))
    print("-----------------------------------")


    '''Question 2: Expected Values'''
    t_x3 = 3
    x3_q2 = []
    j = 0
    for m in range(0, m1):
        x_t = x0
        for i in range(0, d*t_x3):
            x_t = x_t + 1/4*x_t*dt+1/3*x_t*dt_s*w[j] - 3/4*x_t*dt_s*z[j]
            j += 1
        x3_q2.append(x_t)

    E1_q2 = np.mean([(1+x)**(1/3) for x in x3_q2])
    
    t_y3 = 3
    y3_q2 = []
    j = 0
    for m in range(0, m1):
        w2 = 0
        z2 = 0
        for i in range(0, d*t_y3):
            w2 += dt_s*w[j]
            z2 += dt_s*z[j]
            j += 1
        y3_q2.append(np.exp(-0.08*t_y3 + (1/3)*w2 + (3/4)*z2))
    
    E2_q2 = np.mean([(1+y)**(1/3) for y in y3_q2])
    
    ##Question 2 Summary:
    print("For Question 2:")
    print("E1_q2 = " + str(round(E1_q2,6)))
    print("E2_q2 = " + str(round(E2_q2,6)))
    print("-----------------------------------------")
    
    
    '''Question 3: European Call'''
    s0 = np.arange(15,25+1,1)
    t3 = 0.5
    sigma3 = 0.25
    r3 = 0.04
    x3 = 20 #strike price
    
    ##(a) Using Monte Carlo Simulation with Antihetic Variation method
    ##Simulate for 1000 times with intervial 0.01
    c1 = []
    for s in s0:
        c_plus = geoBrownian(n = m1, start = s, t = t3, r = r3,sigma = sigma3, 
                       normal = z, divide = d*t3, single = True)
        c_minus = geoBrownian(n = m1, start = s, t = t3, r = r3,sigma = sigma3, 
                       normal = -np.array(z), divide = d*t3, single = True)
        payoff_plus = [max(c-x3, 0) for c in c_plus]
        payoff_minus = [max(c-x3, 0) for c in c_minus]
        payoff = [(payoff_plus[i] + payoff_minus[i])/2 for i in range(0, m1)]
        c1.append(np.exp(-r3*t3)*np.mean(payoff))
        
    
    ##(b) Black-Scholes Model
    c2 = []
    for s in s0:
        x1 = (np.log(s/x3)+(r3+sigma3**2/2)*t3)/(sigma3*np.sqrt(t3))
        x2 = x1 - sigma3*np.sqrt(t3)
        
        n_x1 = normal_cdf(x1)
        n_x2 = normal_cdf(x2)
        
        c2.append(s*n_x1-x3*np.exp(-r3*t3)*n_x2)
    
    
    ##(c) Greeks
    ##In this question, for simplicity consideration, 
    ##I used the BSM to calculate call price
    delta = []
    gamma = []
    theta = []
    vega = []
    rho = []
    epsilon = 0.005 #I set the epsilon as 0.5% of whatever values it is applying to
    
    for s in s0:
        
        ##Estimate delta
        s_range = s*epsilon*2
        s_plus = s*(1+epsilon)
        s_minus = s*(1-epsilon)
        dd = (bsm(s_plus, x3, r3, t3, sigma3) - bsm(s_minus, x3, r3, t3, sigma3))/s_range
        delta.append(dd)
        
        ##Estimate gamma
        g = ((bsm(s_plus, x3, r3, t3, sigma3)-bsm(s, x3, r3, t3, sigma3)) \
             - (bsm(s, x3, r3, t3, sigma3)-bsm(s_minus, x3, r3, t3, sigma3)))/((s_range/2)**2)
        gamma.append(g)
        
        ##Estimate theta
        dt3 = 1/365/2
        t = (bsm(s, x3, r3, t3+dt3, sigma3) - bsm(s, x3, r3, t3-dt3, sigma3))/(-2*t3)
        theta.append(t)
        
        ##Estimate vega
        v_range = sigma3*epsilon*2
        v_plus = sigma3*(1+epsilon)
        v_minus = sigma3*(1-epsilon)
        v = (bsm(s, x3, r3, t3, v_plus) - bsm(s, x3, r3, t3, v_minus))/v_range
        vega.append(v/100)  
        
        ##Estimate rho
        r_range = r3*epsilon*2
        r_plus = r3*(1+epsilon)
        r_minus = r3*(1-epsilon)
        rr = (bsm(s, x3, r_plus, t3, sigma3) - bsm(s, x3, r_minus, t3, sigma3))/r_range
        rho.append(rr/100)
        
    #Summary of Question 3
    summary_3 = pd.DataFrame([c1,c2, delta, gamma, theta, vega, rho], 
                             columns = [str(s) for s in s0],
                           index = ["Monte-Carlo Price", "BSM Price",
                                    "Delta", "Gamma","Theta", "Vega", "Rho"]).T
    
    plt.figure()
    f, (ax1, ax2, ax3, ax4, ax5)=plt.subplots(5,1, sharex = True)
    ax1.plot(summary_3['Delta'])
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
    
    print("For Question 3: the Index is the stock price")
    print(summary_3)
    print("-----------------------------------------------")

    
    '''Question 4: Heston Model (Stock Price with Volatility)'''
    s0_4 = 48
    k4 = 50
    t4 = 0.5
    rho4 = -0.6
    r4 = 0.03
    v0 = 0.05
    sigma4 = 0.42
    alpha = 5.8
    beta = 0.0625
    
    ##This is to create correlated Wiener process
    dw2 = rho4*np.array(z) + np.sqrt((1-rho4**2))*np.array(w)
    m4 = 5000
    
    ##Full Truncation
    j = 0
    s_4ft = []
    for m in range(0, m4):
        s_t = s0_4
        v_t = v0        
        for i in range(0, int(t4*d)):
            s_t = s_t + r4*s_t*dt + np.sqrt(max(v_t,0))*s_t*dt_s*z[j]
            v_t = v_t + alpha*(beta-max(v_t,0))*dt + sigma4*np.sqrt(max(v_t,0))*dt_s*dw2[j]
            j+=1
        s_4ft.append(s_t)
    
    c1_q4 = np.exp(-r4*t4)*np.mean([max(s-k4, 0) for s in s_4ft])
    
    
    ##Partial Truncation
    j = 0
    s_4pt = []
    for m in range(0, m4):
        s_t = s0_4
        v_t = v0        
        for i in range(0, int(t4*d)):
            s_t = s_t + r4*s_t*dt + np.sqrt(max(v_t,0))*s_t*dt_s*z[j]
            v_t = v_t + alpha*(beta-v_t)*dt + sigma4*np.sqrt(max(v_t,0))*dt_s*dw2[j]
            j+=1
        s_4pt.append(s_t)
    
    c2_q4 = np.exp(-r4*t4)*np.mean([max(s-k4, 0) for s in s_4pt])    
    
    
    ##Reflection
    j = 0
    s_4rf = []
    for m in range(0, m4):
        s_t = s0_4
        v_t = v0        
        for i in range(0, int(t4*d)):
            s_t = s_t + r4*s_t*dt + np.sqrt(abs(v_t))*s_t*dt_s*z[j]
            v_t = abs(v_t) + alpha*(beta-abs(v_t))*dt + sigma4*np.sqrt(abs(v_t))*dt_s*dw2[j]
            j+=1
        s_4rf.append(s_t)
    
    c3_q4 = np.exp(-r4*t4)*np.mean([max(s-k4, 0) for s in s_4rf]) 

    ##Summary:
    summary_4 = pd.DataFrame([c1_q4,c2_q4,c3_q4], columns = ["Call Price"], 
                             index = ["Full Truncation", "Partial Truncation", 
                                      "Reflection Method"])
    
    print("For Question 4: dt = 0.01 and # of simulation is 5000")
    print(summary_4)
    print("---------------------------------------------------------------")
    
    
    '''Question 5: Multi-dimensional random number'''
    ##(a) Generate 2-dimensional uniform random variables
    n5 = 100
    u1 = uniform(size = n5, seed = seed1)
    u2 = uniform(size = n5, seed = seed2)
    
    u_5a = [[u1[i], u2[i]] for i in range(0, n5)]
    
    
    ##(b) Halton sequence with base 2 and 7
    base_5b1 = 2
    base_5b2 = 7
    h_5b1 = halton(size = n5, base = base_5b1)
    h_5b2 = halton(size = n5, base = base_5b2)
    
    
    ##(c) Halton sequence with base 2 and 4
    base_5c1 = 2
    base_5c2 = 4
    h_5c1 = halton(size = n5, base = base_5c1)
    h_5c2 = halton(size = n5, base = base_5c2)    
    
    ##(d) Plot and comment
    plt.figure()
    plt.plot(u1, u2, '.')
    plt.title('Random Uniform Number Generation')
    
    plt.figure()
    plt.plot(h_5b1,h_5b2, '.')
    plt.title('Halton Sequence with base 2 and 7')
    
    plt.figure()
    plt.plot(h_5c1,h_5c2, '.')
    plt.title('Halton Sequence with base 2 and 4')    
    
    
    ##(e) Use Halton to compute integral
    base_e1 = (2,4)
    base_e2 = (2,7)
    base_e3 = (5,7)
    
    n5e = 10000
    
    pairs = [base_e1, base_e2, base_e3]
    integral = []
    for b1, b2 in pairs:
        h1 = halton(size = n5e, base = b1)
        h2 = halton(size = n5e, base = b2)
        fdata = []
        for i in range(0, n5e):
            x = h1[i]
            y = h2[i]
            temp = np.cos(2*np.pi*y)
            if temp > 0:
                temp = temp**(1/3)
            else:
                temp = -(-temp)**(1/3)
            f = np.exp(-x*y)*(np.sin(6*np.pi*x)+temp)
            fdata.append(f)
            
        integral.append(np.mean(fdata))
    
    ##Summary
    summary_5e = pd.DataFrame(integral, columns = ['Integral Estimate'],
                              index = [str(b) for b in pairs])
    
    print('For Question 5e: The integral estimate using Halton sequence with \
          different bases are')
    print(summary_5e)
    print("----------------------------")
    
    