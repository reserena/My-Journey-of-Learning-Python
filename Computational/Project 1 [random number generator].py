# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:58:37 2019

@author: Serena Peng
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

'''This HW is written in Python3.7. It is structured so that the functions 
are on top and then the script follows. 
You can run the whole file at once [F5 (Spyder)]'''

##------------------------function area---------------------------------##
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


def discrete(size, values, probs, uniform):
    '''Generate general discrete with unequal probabilities specified
    by the user. Must include the a set of random uniform variables of
    the same size'''
    
    try:
        if round(sum(probs),6) != 1:
            print("Probabilities do not sum up to 1")
            return None
    except ValueError:
        print("Probabilities cannot sum")
    
    
    if len(values) != len(probs):
        print("Input length inconsistent!")
        return None
    
    if len(uniform) != size:
        print("Lack random uniform variables...Can't perform...")
        return None
    
    try:
        output= []
        for i in uniform:
            for j in range(0, len(values)):
                if i < sum(probs[:j+1]):
                    output.append(values[j])
                    break
                
        return output
    
    except:
        print("Error looping through input")
        return None
            

def binomial(size, n, p, uniform):
    
    if (p >= 1.0) or (n <= 0) or (size <= 0):
        print("Invalid input")
        return None
    
    try:
        if (size*n) != len(uniform):
            print("Not enough uniform variables")
            return None
    except ValueError:
        print("Invalid entry")
     
    j = 0
    output = []
    for i in range(0, size):
        output.append(sum([1 if b <= p else 0 for b in uniform[j:j+n]]))
        j += n
        
    return output
    
    
def exponential(size, lam, uniform):
    
    import math
    
    if (lam <= 0) or (size <= 0):
        raise ValueError("size and lam cannot be less than 0")
        
    if len(uniform) != size:
        print("Insufficient uniform random variables")
        return None
    
    return [-lam*math.log(1-i) for i in uniform]
    
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
        
#------------------------function zone end-----------------------------------#
        

#-----------------------script zone starts-----------------------------------#
if __name__ == '__main__':
    
    seed = 7 #set all seed as 7
    
    ##Question 1: Uniform Distribution
    #(a) LGM Method
    n = 10000
    output_1a = uniform(size = n, seed = seed)
    
    mean_1a = np.mean(output_1a)
    std_1a = np.std(output_1a)
    
    #(b) Using built-in function
    np.random.seed(seed = seed)
    output_1b = np.random.uniform(size = n).tolist()

    mean_1b = np.mean(output_1b)
    std_1b = np.std(output_1b)
    
    #(c) Compare
    summary_1 = pd.DataFrame({'LGM-Method': [mean_1a, std_1a], 
                    'Built-in': [mean_1b, std_1b]}, index = ['Mean', 'Std'])
    
    #Using the output from 1a for the following questions
    u = output_1a.copy()
    
    ##Question 2: Generate Discrete Distribution
    #(a) Generate variables
    values = [-1, 0, 1, 2]
    probs = [0.3, 0.35, 0.2, 0.15]
    
    output_2a = discrete(size = n, values = values, probs = probs, uniform = u)
    
    #(b) Plot and compute mean and standard deviation
    plt.figure()
    bins_2a = np.arange(values[0], values[-1]+1.5)-0.5
    fig, ax = plt.subplots()
    _ = ax.hist(output_2a, bins = bins_2a, rwidth = 0.5)
    ax.set_xticks(bins_2a + 0.5)
    ax.set_xlim(-1.5, 2.5)
    
    mean_2b = np.mean(output_2a)
    std_2b = np.std(output_2a)
    
    ##Question 3: Binomial distribution
    #(a) Generate the variables
    #First, we need to generate 44,000 uniformly distributed random variables
    n3 = 1000
    ber_n = 44
    u3 = uniform(size = n3*ber_n, seed = seed)
    
    p = 0.64
    output_3a = binomial(size = n3, n = ber_n, p = p, uniform = u3)
    
    #(b) Draw the histogram
    plt.figure()
    bins_3a = np.arange(min(output_3a), max(output_3a)+1.5)-0.5
    fig, ax = plt.subplots()
    _ = ax.hist(output_3a, bins = bins_3a, rwidth = 0.5)
    ax.set_xticks(bins_3a + 0.5)
    
    #Empirically calculate P(X>=40)
    prob_3b = sum([1 for i in output_3a if i >= 40])/len(output_3a)

    #using equation to calculate P(X>=40)
    prob_3b_formula = 0
    for k in range(40, 44+1):
        prob_3b_formula += math.factorial(ber_n)/(math.factorial(k)* \
                        math.factorial(ber_n-k))*(p**k)*(1-p)**(ber_n-k)
                        
    prob_summary = (prob_3b, prob_3b_formula)

    ##Question 4: Exponential Distribution
    #(a) Generate variables
    lam = 1.5
    output_4a = exponential(size = n, lam = lam, uniform = u)
    
    #(b) Compute P(X>=1) and P(X>=4)
    #Compute P(X>=1)
    prob_4b1 = sum([1 for i in output_4a if i >= 1])/len(output_4a)
    
    #Compute P(X>=4)
    prob_4b2 = sum([1 for i in output_4a if i >= 4])/len(output_4a)
    
    #(c) Plot and compute mooments
    mean_4b = np.mean(output_4a)
    std_4b = np.std(output_4a)
    
    plt.figure()
    plt.hist(output_4a, bins = 40)
    
    
    ##Question 5: Normal Distribution
    #(a) Generate uniform random variables for later use
    n5 = 5000
    u5 = uniform(size = n5, seed = seed)
    
    #(b) Box-Muller Method
    start_bm = time.time()
    output_5b_bm = normal(size = n5, uniform = u5)
    end_bm = time.time()
    
    #(c) Calculate the mean and std for (b)
    mean_bm = np.mean(output_5b_bm)
    std_bm = np.std(output_5b_bm)
    
    #(d) Polar-Marsaglia method
    output_5d_pm = normal(size = n5, uniform = u5, method = "PM")
    #note: this is less than 4000 variables generated
    
    #(e) Calculate the mean and std for (d)
    mean_pm = np.mean(output_5d_pm)
    std_pm = np.std(output_5d_pm)
    
    summary_normal = pd.DataFrame({"Box-Muller": [str(n5), mean_bm, std_bm],
                                   "Polar-Marsaglia": [str(len(output_5d_pm)),
                                                       mean_pm, std_pm]}, 
                                index = ["n", "Mean", "Standard deviation"])
    
    #(f) Compare the running time
    #To compare the running time, we need to use both methods to generate 
    #same amount of variables. As we know that PM method requires more
    #uniform variables, so I will rerun the PM method here.
    u5f = uniform(size = 30000, seed = seed)
    
    #The reason I limit the uniform size to n5/0.7 is that we can observe 
    #PM method throws out about 22% of the uniform variables. So to generate
    #required number of normally distributed variables, we need at least
    #(size/(1-0.22)), which I approximate to (size/0.7).
    #Also since I want to reuse u5f for different sizes. If the input is 
    #unnecessarily large for the size required, it will slow down
    #the execution. Since I am measuring and comparing the running time, 
    #I don't want the input to skew my running time.
    start_pm = time.time()
    output_5f_pm = normal(size = n5, uniform = u5f[:int(n5/0.7)], method = "PM")
    end_pm = time.time()
    
    running_bm = end_bm - start_bm
    running_pm = end_pm - start_pm
    
    test = [10000, 15000, 20000]
    
    rm = [[running_bm, running_pm]]
    for t in test:
        start1 = time.time()
        output = normal(size = t, uniform = u5f, method = "BM")
        end1 = time.time()
        
        start2 = time.time()
        output = normal(size = t, uniform = u5f[:int(t/0.7)], method = "PM")
        end2 = time.time()
        
        rm.append([end1-start1, end2-start2])
        
        
    rm_summary = pd.DataFrame(rm, columns = ["Box-Muller", "Polar-Marsaglia"],
                              index = ['5000', '10000', '15000', '20000'])
    
