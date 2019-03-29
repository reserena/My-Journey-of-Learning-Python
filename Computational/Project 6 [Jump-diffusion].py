# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:13:40 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''This file is comprised of three sections, new function, old function (from
previous project), and script section;
You can run the whole file at once. 
The expected runtime is about 8-10 minutes'''

#---------------------------new function zone--------------------------------#

def jumpdiffusion(m, start, mu, sigma, lam, n, t, normal, gamma = 1, seed = 3):
    '''Simulate jump-diffusion process
    dv/v = μdt + σdW + γdJ'''
 
    import numpy as np
    para = np.array([m, start, sigma, lam, n, t, seed])
    
    try:
        if (para <= 0).any():
            print("Invalid argument, cannot be negative")
            return None
    except:
        print("Invalid argument type")
        return None
        
    if len(normal) < n*m:
        print("Insufficient normal variables")
        return None
    
    dt = t/n
    
    np.random.seed(seed)
    jump = gamma*np.array(np.random.poisson(lam = lam*dt, size = int(n*m))).reshape((m, n))+1
    j = n-1
    while j != 0:
        jump[:,j] = np.product(jump[:,:j], axis = 1)
        j -= 1
        
    vpaths = np.zeros((m, n+1)) + start
    vpaths[:,1:] = vpaths[:,1:]*jump
    
    trange = np.arange(dt, t+dt, dt)
    normal = normal[:int(m*n)].reshape((m, n))
    w = normal*np.sqrt(dt)# + jump
    del normal ##free up memory
    for d in range(1, n):
        w[:,d] += w[:, int(d-1)]
#        jump[:,d] += jump[:, int(d-1)]
    ft = np.exp((mu-(sigma**2)/2)*trange+sigma*w)# + jump)
    vpaths[:,1:] = vpaths[:,1:]*ft
                
    return vpaths

 
#---------------------------Old function zone--------------------------------#
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


#---------------------------Script zone starts-------------------------------#

if __name__ == '__main__':
    
    seed = 7
    dt = 1/250
    
    '''Question 1: Fixed Strike Lookback Call and Put'''
    r1 = 0.03
    s1 = 98
    k1 = 100
    t1 = 1
    v1 = np.arange(0.12, 0.48+0.01, 0.04)
    m1 = 10000
    
    size = int(t1/dt*m1/2)
    z1 = np.array(normal(size = size, uniform = uniform(size = size, seed = seed)))
    
    call_1 = []
    put_1 = []
    for v in v1:
        paths1 = geoBrownian(n = m1, start = s1, r = r1, sigma = v, t = t1, 
                             normal = np.append(z1, -z1), divide = int(t1/dt))
        s_max = np.amax(paths1, axis = 1)
        s_min = np.amin(paths1, axis = 1)
        
        payoff_c = s_max - k1
        payoff_c[payoff_c < 0] = 0
        call_1.append(np.exp(-r1*t1)*np.mean(payoff_c))
        
        payoff_p = k1 - s_min
        payoff_p[payoff_c < 0] = 0
        put_1.append(np.exp(-r1*t1)*np.mean(payoff_p))
    
    ##Summary
    summary1 = pd.DataFrame([call_1, put_1], columns = v1, index = ['Call', 'Put']).T
    print("Question 1: Fixed Strike Lookback Option")
    print(summary1)
    print("--------------------------------------")
    
    plt.figure()
    plt.plot(v1, call_1)
    plt.title("Fixed Strike Lookback Call Price vs. Volatility")
    
    plt.figure()
    plt.plot(v1, put_1, c = 'g')
    plt.title("Fixed Strike Lookback Put Price vs. Volatility")  
    plt.ylim((8,41))
    
    del paths1
    
    '''Question 2: Jump-Diffusion Model'''
    #Base parameters
    v0 = 20000
    l0 = 22000
    mu = -0.1
    sigma = 0.2
    gamma = -0.4
    lam1r = np.arange(0.05,0.4+0.05, 0.05)
    
    t2 = [3,4,5,6,7,8,9]
    dt2 = 1/12 #monthly compounding
    
    rf2 = 0.02
    lam2r = np.arange(0, 0.8+0.1, 0.1)
    delta = 0.25
    seedr = [11,13,17,19,23,29,31]
    
    alpha = 0.7
    epsilon = 0.95 #recover rate
    
    m2 = 20000 #paths
    
    ##In total of 504 prices
    output2_price = {}
    output2_prob = {}
    output2_etime = {}
    
    import time
    begin = time.time()
    for t in t2:
        n = t*12
        size2 = int(n*m2/2)
        z2 = np.array(normal(size = size2, uniform = uniform(size = size2, seed = t)))

        trange = np.arange(0, t+0.01, dt2)
        beta = (epsilon - alpha)/t  
        
        for lam1 in lam1r:            
            ##Antithetic variation
            path_v = jumpdiffusion(m = m2, start = v0, mu = mu, sigma = sigma,
                            lam = lam1, n = n, t = t, normal = np.append(z2,-z2),
                            gamma = gamma, seed = seedr[-3+t])
            
            for lam2 in lam2r:
                r2 = (rf2 + delta*lam2)/12
         
                pmt = l0*r2/(1-1/((1+r2)**n))
                a = pmt/r2
                b = pmt/(r2*((1+r2)**n))
                c = (1+r2)
                
                ##Default Q:
                qt = alpha + beta*trange[1:]
                path_l = a-b*(c**(12*trange[1:]))
                path_l[n-1] = 0
                qmark = np.repeat(qt*path_l, repeats = m2).reshape((n,m2)).T
                qmark = qmark - path_v[:,1:]
                      
                ##Default S:
                np.random.seed(t)
                nt = np.array(np.random.poisson(lam = lam2*dt2, 
                                                size = int(m2*n))).reshape((m2, n))
                
                payoff = np.array([])
                tao = np.array([])
                for i in range(0, m2):
                    try:
                        sss = np.where(nt[i,:] > 0)[0][0]
                    except:
                        sss = n+1 ##no jump
                        
                    try:
                        qqq = np.where(qmark[i,:] > 0)[0][0]
                    except:
                        qqq = n+1
                        
                    if min(sss,qqq) == n+1:
                        pf = 0 #if no exercise, the default option payoff = 0
                    elif qqq < sss:
                        #if exercised, the payoff = e^(-rt)*payoff
                        pf = np.exp(-rf2*dt2*(qqq+1))*max(path_l[qqq] - \
                                   epsilon*path_v[i,qqq+1],0)
                    else:
                        pf = np.exp(-rf2*dt2*(sss+1))*abs(path_l[sss] - \
                                   epsilon*path_v[i,sss+1])
                    
                    tao = np.append(tao, min(qqq, sss))
                    payoff = np.append(payoff,pf)
                    
                price = np.mean(payoff)
                output2_price['t{}_lam1={}_lam2={}'.format(t, lam1, lam2)] = price
                output2_prob['t{}_lam1={}_lam2={}'.format(t, lam1, lam2)] = 1-len(tao[tao==n+1])/m2
                output2_etime['t{}_lam1={}_lam2={}'.format(t, lam1, lam2)] = np.mean(tao[tao!=n+1])+1
                    
                
    end = time.time()
    
    ##Summary
    d2 = output2_price['t5_lam1=0.2_lam2=0.4']
    prob = output2_prob['t5_lam1=0.2_lam2=0.4']
    et = output2_etime['t5_lam1=0.2_lam2=0.4']

    pfull = list(output2_price.items())
    pindex = [p[0].split('_')[1:] for p in pfull]
    index = np.array([[x[:9], y[:9]] for x, y in pindex][:72]).T.tolist() #clean the index
    price = pd.DataFrame(np.array([p[1] for p in pfull]).reshape((7,72)).T, 
                         columns = t2).set_index(index, drop = True)
    
    probfull = list(output2_prob.items())
    probs = pd.DataFrame(np.array([p[1] for p in probfull]).reshape((7,72)).T, 
                         columns = t2).set_index(index, drop = True)
    
    etfull = list(output2_etime.items())
    etimes = pd.DataFrame(np.array([p[1] for p in etfull]).reshape((7,72)).T, 
                         columns = t2).set_index(index, drop = True)
    
    ##plot1: when lam1 is set at 0.2
    plot1p = price.loc[('lam1=0.2',)]
    plot1prob = probs.loc[('lam1=0.2',)]
    plot1et = etimes.loc[('lam1=0.2',)]
    
    plt.figure()
    plt.plot(plot1p.T)
    plt.legend(plot1p.index)
    plt.xlabel('T')
    plt.ylabel('Price')
    plt.title('Option Price as a function of T (λ1 = 0.2)')
    
    plt.figure()
    plt.plot(plot1prob.T)
    plt.legend(plot1prob.index)
    plt.xlabel('T')
    plt.ylabel('Default Prob')
    plt.title('Default Probability as a function of T (λ1 = 0.2)')    

    plt.figure()
    plt.plot(plot1et.T)
    plt.legend(plot1et.index)
    plt.xlabel('T')
    plt.ylabel('Expected Exercise Time (months)')
    plt.title('Expected Exercise Time as a function of T (λ1 = 0.2)')
    
    ##plot2: when lam2 is set at 0.4
    plot2p = price.reorder_levels([1,0]).sort_index().loc[('lam2=0.4',)]
    plot2prob = probs.reorder_levels([1,0]).sort_index().loc[('lam2=0.4',)]
    plot2et = etimes.reorder_levels([1,0]).sort_index().loc[('lam2=0.4',)]
    
    plt.figure()
    plt.plot(plot2p.T)
    plt.legend(plot2p.index)
    plt.xlabel('T')
    plt.ylabel('Price')
    plt.title('Option Price as a function of T (λ2 = 0.4)')
    
    plt.figure()
    plt.plot(plot2prob.T)
    plt.legend(plot2prob.index)
    plt.xlabel('T')
    plt.ylabel('Default Prob')
    plt.title('Default Probability as a function of T (λ2 = 0.4)')    

    plt.figure()
    plt.plot(plot2et.T)
    plt.legend(plot2et.index, loc = 2)
    plt.xlabel('T')
    plt.ylabel('Expected Exercise Time (months)')
    plt.title('Expected Exercise Time as a function of T (λ2 = 0.4)')    
    
    
    print("Question 6:")
    print("At λ1 = 0.2, λ2 = 0.4, and T = 5, the default option is worth\
          {} with the default probability of {} and expected exercise of \
          about {} months".format(round(d2,6),prob,round(et,2)))
    
    
    
    
