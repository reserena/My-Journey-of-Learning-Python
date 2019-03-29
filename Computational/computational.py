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

def binomialTree(s, k, rf, t, n, u, d, p, option = "call", style = "European", 
                 tree = False):
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


def trinomialTree(s, k, rf, t, n, u, d, p, log = False, option = "call", 
                  style = "European", tree = False):
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


class finitediff:
    '''Finite-Difference Method for option pricing, x = ln(s)'''
    
    def __init__(self, drange, dx, k, sigma, r, t, dt, on = 'x', 
                 method = 'EFD', option = 'call', style = 'European'):
        self.drange = np.array(sorted(drange, reverse = True))
        self.dx = dx
        self.k = k
        self.v = sigma
        self.r = r
        self.t = t
        self.dt = dt
        self.on = on
        self.method = method
        self.option = option
        self.style = style
        
    def model(self):
        
        para = np.array([self.dx, self.k, self.v, self.r, self.t, self.dt])
        
        try:
            if (para < 0).any():
                print("Invalid negative input")
                return None
        except:
            print('Input args are not all numberic')
            return None
        
        #xrange = sorted(self.drange, reverse = True)
        
        n = len(self.drange)
        m = int(self.t/self.dt)
        
        
        smodel = np.repeat(self.drange, repeats = m).reshape((n, m))
        if self.on == 'x':
            smodel = np.exp(smodel)
        

        if self.option == 'call':
            fff = smodel - self.k
        elif self.option == 'put':
            fff = self.k- smodel 
        else:
            print('Invalid option argument')
            return None
        
        fff[fff < 0] = 0
        
        if self.method == 'EFD':
            return self.__EFD(n, m, fff, smodel)
        elif self.method == 'IFD':
            return self.__IFD(n, m, fff, smodel)
        elif self.method == 'CNFD':
            return self.__CNFD(n, m, fff, smodel)
        else:
            print('Invalid method')
            return None
        
    def __EFD(self, n, m, fff, smodel):
        '''Explicit Finite Difference Method'''
        
        if self.on == 'x':  
            p_u = self.dt*((self.v**2)/(2*(self.dx**2))+(self.r-(self.v**2)/2)/(2*self.dx))
            p_m = 1 - self.dt*(self.v**2)/(self.dx**2) - self.r*self.dt
            p_d = self.dt*((self.v**2)/(2*(self.dx**2))-(self.r-(self.v**2)/2)/(2*self.dx))
        elif self.on == 's':
            jjj = self.drange[1:n-1]/self.dx
            
            p_u = self.dt*((self.r*jjj)/2+(self.v**2)*(jjj**2)/2)
            p_m = 1 - self.dt*((self.v**2)*(jjj**2)+self.r)
            p_d = self.dt*(-(self.r*jjj)/2+(self.v**2)*(jjj**2)/2)
        else:
            print("Invalid on argument")
            return None
            
        aaa = np.zeros((n,n))
        np.fill_diagonal(aaa[1:n-1,:n-2],p_u)
        np.fill_diagonal(aaa[1:n-1,1:n-1], p_m)
        np.fill_diagonal(aaa[1:n-1,2:], p_d)
        aaa[0,:3] = aaa[1,:3]
        aaa[n-1,n-3:] = aaa[n-2,n-3:]
            
        bbb = np.zeros((1,n))
        if self.option == 'call':
            bbb[0,0] = smodel[0,0]-smodel[1,0]
        elif self.option == 'put':
            bbb[0,n-1] = -(smodel[n-1,0]-smodel[n-2,0])
            
        if self.style == "European":
            for c in reversed(range(0,m-1)):
                fff[:,c] = aaa@fff[:,c+1] + bbb
        elif self.style == "American":
            for c in reversed(range(0,m-1)):
                value = aaa@fff[:,c+1] + bbb   
                fff[np.where(value > fff[:,c])[1],c] = value[value > fff[:,c]]           
        else:
            print('Invalid style argument')
            return None
        
        return fff[:,0]        

    def __IFD(self, n, m, fff, smodel):
        '''Implicit Finite Difference Method'''
        
        if self.on == 'x': 
            p_u = -0.5*self.dt*((self.v**2)/(self.dx**2)+(self.r-(self.v**2)/2)/(self.dx))
            p_m = 1 + self.dt*(self.v**2)/(self.dx**2) + self.r*self.dt
            p_d = -0.5*self.dt*((self.v**2)/(self.dx**2)-(self.r-(self.v**2)/2)/(self.dx))
        elif self.on == 's':
            jjj = self.drange[1:n-1]/self.dx
            
            p_u = -0.5*self.dt*(self.r*jjj+(self.v**2)*(jjj**2))
            p_m = 1 + self.dt*((self.v**2)*(jjj**2)+self.r)
            p_d = -0.5*self.dt*(-(self.r*jjj)+(self.v**2)*(jjj**2))            
        else:
            print("Invalid on argument")
            return None

        aaa = np.zeros((n,n))
        aaa[0,:2] = np.array([1,-1])
        aaa[n-1,n-2:] = np.array([1,-1])
        np.fill_diagonal(aaa[1:n-1,:n-2],p_u)
        np.fill_diagonal(aaa[1:n-1,1:n-1], p_m)
        np.fill_diagonal(aaa[1:n-1,2:], p_d) 
        
        ainv = np.linalg.inv(aaa)

        bb = np.zeros((1,n))
        if self.option == 'call':
            bb[0,0] = smodel[0,0]-smodel[1,0]
        elif self.option == 'put':
            bb[0,n-1] = (smodel[n-1,0]-smodel[n-2,0])
                
        if self.style == "European":
            for c in reversed(range(0,m-1)):
                bbb = bb.copy()
                bbb[0,1:n-1] = fff[1:n-1,c+1]
                ab = ainv@np.transpose(bbb)
                fff[:,c] = ab[:,0]   
        elif self.style == "American":
            for c in reversed(range(0,m-1)):
                bbb = bb.copy()
                bbb[0,1:n-1] = fff[1:n-1,c+1]
                ab = ainv@np.transpose(bbb)
                fff[np.where(ab[:,0] > fff[:,c])[0],c] = ab[ab[:,0] > fff[:,c],0]

        return fff[:,0]
    
    def __CNFD(self, n, m, fff, smodel):
        
        if self.on == 'x': 
            p_u = -(1/4)*self.dt*((self.v**2)/(self.dx**2)+(self.r-(self.v**2)/2)/(self.dx))
            p_m = 1 + self.dt*(self.v**2)/(2*(self.dx**2)) + (self.r*self.dt/2)
            p_d = -(1/4)*self.dt*((self.v**2)/(self.dx**2)-(self.r-(self.v**2)/2)/(self.dx))
        elif self.on == 's':
            jjj = self.drange[1:n-1]/self.dx
            
            p_u = -(1/4)*self.dt*((self.v**2)*(jjj**2)+self.r*jjj)
            p_m = 1 + 0.5*self.dt*((self.v**2)*(jjj**2) + self.r/2)
            p_d = -(1/4)*self.dt*((self.v**2)*(jjj**2)-self.r*jjj)
        else:
            print("Invalid on argument")
            return None
            
        aaa = np.zeros((n,n))
        aaa[0,:2] = np.array([1,-1])
        aaa[n-1,n-2:] = np.array([1,-1])
        np.fill_diagonal(aaa[1:n-1,:n-2],p_u)
        np.fill_diagonal(aaa[1:n-1,1:n-1], p_m)
        np.fill_diagonal(aaa[1:n-1,2:], p_d)
            
        ppp = np.zeros((n-2,n))
        np.fill_diagonal(ppp[:,:n-2], -p_u)
        np.fill_diagonal(ppp[:,1:n-1], -(p_m-2))
        np.fill_diagonal(ppp[:,2:n], -p_d)
    
        ainv = np.linalg.inv(aaa)

        zz = np.zeros((1,n))
        if self.option == 'call':
            zz[0,0] = smodel[0,0]-smodel[1,0]
        elif self.option == 'put':
            zz[0,n-1] = (smodel[n-1,0]-smodel[n-2,0])
                
        if self.style == "European":
            for c in reversed(range(0,m-1)):
                zzz = zz.copy()
                zzz[0,1:n-1] = ppp@fff[:,c+1]
                ab = ainv@np.transpose(zzz)
                fff[:,c] = ab[:,0]
        elif self.style == "American":
            for c in reversed(range(0,m-1)):
                zzz = zz.copy()
                zzz[0,1:n-1] = ppp@fff[:,c+1]
                ab = ainv@np.transpose(zzz)
                fff[np.where(ab[:,0] > fff[:,c])[0],c] = ab[ab[:,0] > fff[:,c],0]            
                
        return fff[:,0]

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

def numerix(pv0, wac, t, dt, rmodel, m = 1, r10 = None):
    '''Use numerix prepayment model to calculate conditional 
    prepayment rate (CPR), which is then used to value mortgage
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



##############################################################################
#-----------------------------less often used--------------------------------#
##############################################################################
        
'''Old normal function'''
def normal2(size, uniform, method = "BM", loop = False):
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


def discrete(size, values, probs, uniform):
    '''Generate general discrete with unequal probabilities specified
    by the user. Must include the a set of random uniform variables of
    the same size'''
    
    try:
        if round(sum(probs),6) != 1.0:
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
    '''Generate binomial random variables'''
    
    if (p >= 1.0) or (p <= 0 ) or (n <= 0) or (size <= 0):
        print("Invalid input")
        return None
    
    try:
        if (size*n) != len(uniform):
            print("Not enough uniform variables")
            return None
    except ValueError:
        print("Invalid entry")
    
#    j = 0
#    output = []
#    for i in range(0, size):
#        output.append(sum([1 if b <= p else 0 for b in uniform[j:j+n]]))
#        j += n

    try:
        u = np.array(uniform)
    except:
        print('Cannot reshape uniform variables, need np.darray format')
        
    u[u <= p] = 1
    u[u != 1] = 0
        
    output = u.reshape((size, n)) 
    output = np.sum(output, axis = 1)
    
    return output

def exponential(size, lam, uniform):
    '''Generate exponentially distributed random variables'''
    
    import math
    
    if (lam <= 0) or (size <= 0):
        raise ValueError("size and lam cannot be less than 0")
        
    if len(uniform) != size:
        print("Insufficient uniform random variables")
        return None
    
    u = np.array(uniform)
#    return [-lam*math.log(1-i) for i in uniform]
    return -lam*math.log(1-u)
