# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:08:24 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

'''For this project, there are three sections
First section: Reused functions from Project 1 for random variables generation.
Second section: New functions created for this project;
Last section: Script;
You can run the whole file at once'''

'''2/19/2019 Update: Revise the functino geoBrownian()
change the output data from df to narray, which improves the running time by 90%
Please see the copy of the original function in last section of Project 2.py'''

#--------------------------Function from Project 1---------------------------#

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

#------------------------Function from Project 1 ends------------------------#
        
    
#-----------------------------New Functions----------------------------------#

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

#-----------------------------New Function Ends------------------------------#

#---------------------------Script Zone Starts-------------------------------#

if __name__ == '__main__':
    seed = 7
    
    '''Question 1: Bi-variate Normal'''
    n1 = 2000
    u1 = uniform(size = n1, seed = seed)
    z1 = normal(size = n1, uniform = u1)
    
    a = -0.7
    mu = [0,0]
    
    #Given the relationship var(Y) = a^2 + b^2 = 1 and a = -0.7
    #We can calculate b = sqrt(1-a^2)
    b = np.sqrt(1-a**2)
    output_1x = []
    output_1y = []
    for i in range(0, n1, 2): #step = 2
        x = mu[0] + a*z1[i]
        y = mu[1] + a*z1[i] + b*z1[i+1]
        
        output_1x.append(x)
        output_1y.append(y)
    
    output_1 = (output_1x, output_1y)
    ##Output: rho
    output_1_rho = np.cov(output_1)[0,1]/(np.std(output_1x)*np.std(output_1y))
    print('Question 1: ρ=' + str(round(output_1_rho, 6)))
    
    '''Question 2: Expected value using Monte Carlo Simulation'''
    n2 = 10000
    u2 = uniform(size = n2, seed = seed)
    z2 = normal(size = n2, uniform = u2)
    rho2 = 0.6
    rho2_ = np.sqrt(1-rho2**2) 
    
    output_2 = []
    half = int(n2/2)

    for i in range(0, half): 
        x = z2[i]
        y = rho2*z2[i] + rho2_*z2[i+half]
        
        f = x**3 + np.sin(y) + x**2*y

        output_2.append(max(0, f))
    
    expected_2 = np.mean(output_2)
    print('Question 2: E='+str(round(expected_2, 6)))
    
    '''Question 3: More Simulation'''
    ##3(a) Calculate expected values
    ##Borrowing the Z values from Question 2
    #Assumption: we set n = 1 in sqrt(T/n) when calculate W_T
    n = 1
    #3a.1 E(W5^2+sin(W5))
    output_3a1 = []
    T_3a1 = 5
    for z in z2:
        w5 = np.sqrt(T_3a1/n)*z
        output_3a1.append(w5**2+np.sin(w5))
        
    Ea1 = np.mean(output_3a1)
    Ea1_var = np.var(output_3a1)
    
    #3a.2 E(...) when t = 0.5, 3.2, and 6.5
    T_3a2 = 0.5
    T_3a3 = 3.2
    T_3a4 = 6.5
    output_3a2 = []
    output_3a3 = []
    output_3a4 = []
    for z in z2:
        wt1 = np.sqrt(T_3a2/n)*z
        wt2 = np.sqrt(T_3a3/n)*z
        wt3 = np.sqrt(T_3a4/n)*z
        
        output_3a2.append(np.e**(T_3a2/2)*np.cos(wt1))
        output_3a3.append(np.e**(T_3a3/2)*np.cos(wt2))
        output_3a4.append(np.e**(T_3a4/2)*np.cos(wt3))
        
    Ea2 = np.mean(output_3a2)
    Ea2_var = np.var(output_3a2)
    Ea3 = np.mean(output_3a3)
    Ea3_var = np.var(output_3a3)
    Ea4 = np.mean(output_3a4)
    Ea4_var = np.var(output_3a4)
    
    ##3(b) Compare the last three results in (a)
    ##They are roughly the same. The write-up is in pdf
    
    
    ##3(c) Using variance reduction technique
    ##I choose the Control Variate method
    output_3c1 = []
    output_3c2 = []
    output_3c3 = []
    output_3c4 = []
    
    #For the first equation, the control Y = w^2
    w5 = [np.sqrt(T_3a1)*z for z in z2]
    control_w5 = [w**2 for w in w5]
    mean_w5 = 5 #by numerical calculation
    var_w5 = 50 #by numerical calculation
    gamma_c1 = np.cov(np.stack((output_3a1, control_w5)))[1,0]/var_w5
    output_3c1 = [output_3a1[i] - gamma_c1*(control_w5[i] - mean_w5) for i in range(0, n2)]
    Eb1 = np.mean(output_3c1)
    Eb1_var = np.var(output_3c1)
    
    #For the second equation, the control Y = z^2
    control_y = [z**2 for z in z2]
    mean_y = 1 #by numerical calculation
    var_y = 2 #by numerical calculation
    gamma_c2 = np.cov(np.stack((output_3a2, control_y)))[1,0]/var_y
    output_3c2 = [output_3a2[i] - gamma_c2*(control_y[i] - mean_y) for i in range(0, n2)]
    Eb2 = np.mean(output_3c2)
    Eb2_var = np.var(output_3c2)
    
    gamma_c3 = np.cov(np.stack((output_3a3, control_y)))[1,0]/var_y
    output_3c3 = [output_3a3[i] - gamma_c3*(control_y[i] - mean_y) for i in range(0, n2)]
    Eb3 = np.mean(output_3c3)
    Eb3_var = np.var(output_3c3)    
    
    gamma_c4 = np.cov(np.stack((output_3a4, control_y)))[1,0]/var_y
    output_3c4 = [output_3a4[i] - gamma_c4*(control_y[i] - mean_y) for i in range(0, n2)]
    Eb4 = np.mean(output_3c4)
    Eb4_var = np.var(output_3c4)
    
    #Summary of Question 3
    print('Problem 3: E[x] Before and After Variance Reduction Technique')
    summary_3c = pd.DataFrame([[Ea1, Eb1],[Ea2, Eb2],[Ea3, Eb3],[Ea4, Eb4]],
                            columns = ["W/t Variance Reduction", 
                                       "W/ Variance Reduction"],
                            index = ["W5", "t=0.5", "t=3.2", "t=6.5"])
    print(summary_3c)
    
    summary_3c_var = pd.DataFrame([[Ea1_var, Eb1_var],[Ea2_var, Eb2_var],
                                   [Ea3_var, Eb3_var],[Ea4_var, Eb4_var]],
                            columns = ["W/t Variance Reduction", 
                                       "W/ Variance Reduction"],
                            index = ["W5", "t=0.5", "t=3.2", "t=6.5"])
    
    '''Question 4:European Call'''
    ##4(a) Using simulation to calculate c
    r4 = 0.04
    sigma4 = 0.2
    s0 = 88 #Stock price at t=0
    t4 = 5
    x4 = 100 #Strike price
    
    ##Borrowing the {z} from question 2
    s_t4 = geoBrownian(n = n2, start = s0, t = t4, r = r4, 
                       sigma = sigma4, normal = z2)
    
    payoff_4a = [max(0, s-x4) for s in s_t4]
        
    Ca1 = np.e**(-r4*t4)*np.mean(payoff_4a)
    Ca1_var = (np.e**(-r4*t4))**2*np.var(payoff_4a)
    
    ##4(b) Using BSM to calculate c
    d1 = (np.log(s0/x4)+(r4+sigma4**2/2)*t4)/(sigma4*np.sqrt(t4))
    d2 = d1 - sigma4*np.sqrt(t4)
    
    Cb1 = st.norm.cdf(d1)*s0 - st.norm.cdf(d2)*np.e**(-r4*t4)*x4
    
    ##Using variance reduction technique on (a)
    ##For this question, I use the Antithetic Variation method
    ##I use z and -z to simulate s_t
    z_inv = [-z for z in z2]
    s_t4_inv = geoBrownian(n = n2, start = s0, t = t4, r = r4, 
                       sigma = sigma4, normal = z_inv)    
    payoff_4b = [max(0, s-x4) for s in s_t4_inv]
        
    payoff_4 = [(payoff_4a[i] + payoff_4b[i])/2 for i in range(0, n2)]
    Ca2 = np.e**(-r4*t4)*np.mean(payoff_4)
    Ca2_var = (np.e**(-r4*t4))**2*np.var(payoff_4)
    
    ##We can observe the accuracy does improve
    print('Question 4: Call price is '+str(round(Cb1,6))+' using Black-Scholes model')
    summary_4 = pd.DataFrame({'E[X]': [Ca1, Ca2], 'Variance': [Ca1_var, Ca2_var]},
                             index = ['Monte-Carlo', 'Variance Reduction'])
    print(summary_4)
    
    '''Question 5: Simulate the whole path'''
    ##5(a) Generate S_n for different n from 1 to 10
    r5 = 0.04
    sigma5 = 0.18
    s0 = 88
    n5 = 1000
    
    j = 0
    es_n = []
    n_range = list(range(1,11))
    for n in n_range:
        sim = geoBrownian(n = n5, start = s0, t = n, r = r5,
                          sigma = sigma5, normal = z2[j:j+n5])
        j += n5
        es_n.append(np.mean(sim))
        
    plt.figure()
    plt.bar(n_range, es_n)
    plt.ylabel('S_t')
    plt.xlabel('n')
    plt.xticks(n_range)


    ##5(b) Simluate 6 paths
    path = 6
    divide = 1000
    t6 = 10
    
    output_5b = geoBrownian(n = path, start = s0, t = t6, r = r5,
                          sigma = sigma5, normal = z2, divide = divide)

    ##5(c): plot (a) and (b) together
    x_range = np.arange(0, t6+t6/divide, t6/divide)
    
    plt.figure()
    plt.bar(n_range, es_n)
    plt.ylabel('S_t')
    plt.xlabel('T')
    plt.xticks(n_range)
    plt.plot(x_range, pd.DataFrame(output_5b).T)
    plt.title('Question 5c: σ = 0.18')
    
    ##5(d): write-up in pdf
    
    '''Question 6: Approximate π'''
    ##6(a) Using Euler's Scheme
    step = 0.001 #interval
    n6 = np.arange(0, 1, step)
    
    output_6a = []
    x = 0
    for n in n6:
        x = x + step*np.sqrt(1-n**2)*4
    
    Ia = x
    
    ##6(b) Using Monte-Carlo Simulation
    ##Borrowing the uniform variables from Question 2
    output_6b = [4*np.sqrt(1-u**2) for u in u2]

    Ib = np.mean(output_6b)
    Ib_var = np.var(output_6b)
    
    ##6(c) Importance Sampling to improve (b)
    alpha = 0.76
    t_x = [(1-u*u*alpha)/(1-alpha/3) for u in u2]
    
    output_6c = [output_6b[i]*1/t_x[i] for i in range(0, n2)]
    
    Ic = np.mean(output_6c)
    Ic_var = np.var(output_6c)
    
    print('Question 6:')
    summary_6c = pd.DataFrame({'E[X]': [Ia, Ib, Ic], 'Variance': ['n/a',Ib_var, Ic_var]},
                               index = ['Euler Scheme', 'Monte-Carlo', 
                                        'Variance Reduction'])
    print(summary_6c)


#-----------------------Old versions of functions----------------------------#
    
#def geoBrownian(n, start, r, sigma, t, normal, divide = 1, single = False):
#    '''Simulate Geometric Brownian Motion process S_t
#    n refers to the 
#    When divide = 1, it will generate n S_t;
#    When divide !=1, it will generate n paths of S_t
#    When single == True, only return the last values for each path generated'''
#    
#    if (n <= 0) or (type(n) != int):
#        print("Invalid size input")
#        return None
#    
#    if (sigma < 0) or (start < 0) or (t < 0):
#        print("Invalid negative input")
#        return None
#        
#    if len(normal) < n*divide:
#        print("Insufficient normal variables")
#        return None
#    
#    if (type(divide) != int) & (type(divide) != float):
#        print("Invalid divide input")
#        return None
#    
#    output = []
#    
#    if divide == 1:
#        
#        for i in range(0, n):
#            s_t = start*np.e**((r-sigma**2/2)*t+sigma*np.sqrt(t)*normal[i])
#            output.append(s_t)
#            
#        return output
#    elif divide > 1:
#        tint = t/divide
#        z = 0
#        for i in range(0, n):
#            path = [start]
#            for d in range(0, int(divide)):
#                w = np.sqrt(tint)*normal[z]
#                s_t = path[d]*np.exp((r-(sigma**2)/2)*tint+sigma*w)
#                path.append(s_t)
#                z += 1
#            if single == False:
#                output.append(path)
#            else:
#                output.append(s_t)
#        
#        return output
#    else:       
#        print("Divde cannot be less than 1")
#        return None
