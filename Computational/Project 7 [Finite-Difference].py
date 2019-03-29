# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 06:48:50 2019

@author: Serena Peng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------------new function section---------------------------------#

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
         

#------------------------old function section--------------------------------#

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



#-----------------------script section starts--------------------------------#

if __name__ == '__main__':
    
    '''Question 1: European Put'''
    s0 = 10
    x0 = np.log(s0)
    v1 = 0.2
    r1 = 0.04
    dt1 = 0.002
    k1 = 10
    t1 = 0.5
    s1l = 4
    s1u = 16
    x1l = np.log(s1l)
    x1u = np.log(s1u)
    dxr = v1*np.sqrt(np.array([1,3,4])*dt1)
    m = int(t1/dt1)
    
    summary_1 = [] 
    for dx in dxr:
        xrange = np.append(np.flip(np.arange(x0,x1l,-dx)),np.arange(x0+dx, x1u, dx))
        
        '''1a. Explicit Finite-Difference Method'''
        pa = finitediff(xrange, dx =dx, k = k1, sigma = v1, r = r1, t = t1,
                        dt = dt1, option = 'put').model()
        
        '''1b. Implicit Finite-Difference Method'''
        pb = finitediff(xrange, dx =dx, k = k1, sigma = v1, r = r1, t = t1, 
                        dt = dt1, method = 'IFD', option = 'put').model()
        
        '''1c. Crank-Nicolson Finite-Difference Method'''
        pc = finitediff(xrange, dx =dx, k = k1, sigma = v1, r = r1, t = t1, 
                        dt = dt1, method = 'CNFD', option = 'put').model()
        bsmlst = []
        for s in np.exp(xrange):
            bsmlst.append(bsm(s = s, k = k1, r = r1, t = t1, sigma = v1, option = 'put'))
            
        
        summary_1.append([np.exp(xrange), np.flip(pa), np.flip(pb), np.flip(pc), 
                          bsmlst])
        
        
    '''Summary of Question 1'''
    ##when dx = sigma*sqrt(dt)
    summary1_dx1 = pd.DataFrame(summary_1[0]).T.set_index(0, drop =True)
    summary1_dx1.columns = ['EFD', 'IFD', 'CNFD', 'BSM']    
    summary1_dx1['EFD_relative error'] = 1-summary1_dx1['EFD']/summary1_dx1['BSM']
    summary1_dx1['IFD_relative error'] = 1-summary1_dx1['IFD']/summary1_dx1['BSM']
    summary1_dx1['CNFD_relative error'] = 1-summary1_dx1['CNFD']/summary1_dx1['BSM']
    
    plt.figure()
    plt.plot(summary1_dx1.iloc[:,:4])
    plt.legend(summary1_dx1.iloc[:,:4].columns)
    plt.title('European Put (dx = σ√Δt)')
    
    ##when dx = sigma*sqrt(3dt)
    summary1_dx2 = pd.DataFrame(summary_1[1]).T.set_index(0, drop =True)
    summary1_dx2.columns = ['EFD', 'IFD', 'CNFD', 'BSM']    
    summary1_dx2['EFD_relative error'] = 1-summary1_dx2['EFD']/summary1_dx2['BSM']
    summary1_dx2['IFD_relative error'] = 1-summary1_dx2['IFD']/summary1_dx2['BSM']
    summary1_dx2['CNFD_relative error'] = 1-summary1_dx2['CNFD']/summary1_dx2['BSM']
    
    plt.figure()
    plt.plot(summary1_dx2.iloc[:,:4])
    plt.legend(summary1_dx2.iloc[:,:4].columns)
    plt.title('European Put (dx = σ√3Δt)')
    
    ##when dx = sigma*sqrt(4dt)
    summary1_dx3 = pd.DataFrame(summary_1[2]).T.set_index(0, drop =True)
    summary1_dx3.columns = ['EFD', 'IFD', 'CNFD', 'BSM']    
    summary1_dx3['EFD_relative error'] = 1-summary1_dx3['EFD']/summary1_dx3['BSM']
    summary1_dx3['IFD_relative error'] = 1-summary1_dx3['IFD']/summary1_dx3['BSM']
    summary1_dx3['CNFD_relative error'] = 1-summary1_dx3['CNFD']/summary1_dx3['BSM']
    
    plt.figure()
    plt.plot(summary1_dx3.iloc[:,:4])
    plt.legend(summary1_dx3.iloc[:,:4].columns)
    plt.title('European Put (dx = σ√4Δt)')
    
    pa1ind = np.where(np.array(summary1_dx1.index).round(6) == s0)[0][0]
    pa1 = summary1_dx1.iloc[pa1ind, :3]

    pb1ind = np.where(np.array(summary1_dx2.index).round(6) == s0)[0][0]
    pb1 = summary1_dx2.iloc[pb1ind, :3]
    
    pc1ind = np.where(np.array(summary1_dx3.index).round(6) == s0)[0][0]
    pc1 = summary1_dx3.iloc[pc1ind, :3]
    
    summaryp1 = pd.DataFrame([pa1, pb1, pc1], index = ['σ√1Δt','σ√3Δt','σ√4Δt'])
    
    print('Question 1: European Put Price when s = 10 is')
    print(summaryp1)
    print('------------------------------------------')
    
#    writer = pd.ExcelWriter('question1.xlsx')
#    summary1_dx1.to_excel(writer, 'dx1')
#    summary1_dx2.to_excel(writer, 'dx2')
#    summary1_dx3.to_excel(writer, 'dx3')
#    writer.save()    
    
    '''Question 2: American Call & Put'''    
    k2 = 10
    r2 = 0.04
    v2 = 0.2
    t2 = 0.5
    dt2 = 0.002
    dsr = np.array([0.25, 1, 1.25])
    s2l = 4
    s2h = 16
    
    m = int(t2/dt2)
    
    output2 = []
    for ds in dsr:    

        srange = np.append(np.flip(np.arange(s0,s2l-0.5*ds,-ds)),np.arange(s0+ds,s2h+0.5*ds, ds))
    
        '''2a. Explicit Finite-Difference Method'''
        ca2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', option = 'call', style = "American").model()
        pa2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', option = 'put', style = "American").model()
    
        '''2b. Implicit Finite-Difference Method'''
        cb2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', method = 'IFD', option = 'call', 
                        style = "American").model()
        pb2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', method = 'IFD', option = 'put', 
                        style = "American").model()
        
        '''2c. Crank-Nicolson Finite-Difference Method'''
        cc2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', method = 'CNFD', option = 'call', 
                        style = "American").model()
        pc2 = finitediff(srange, dx = ds, k = k2, sigma = v2, r = r2, t = t2, 
                        dt = dt2, on = 's', method = 'CNFD', option = 'put', 
                        style = "American").model()
        
        output2.append([[np.flip(srange), ca2, cb2, cc2], 
                         [np.flip(srange), pa2, pb2, pc2]])
        
    '''Summary of Question 2'''
    #when ds = 0.25
    summary2_c1 = pd.DataFrame(output2[0][0]).T.set_index(0, drop =True)
    summary2_c1.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_c1)
    plt.legend(summary2_c1.columns)
    plt.title('American call (ds = 0.25)')
    
    summary2_p1 = pd.DataFrame(output2[0][1]).T.set_index(0, drop =True)
    summary2_p1.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_p1)
    plt.legend(summary2_p1.columns)
    plt.title('American put (ds = 0.25)')   
    
    #when ds = 1
    summary2_c2 = pd.DataFrame(output2[1][0]).T.set_index(0, drop =True)
    summary2_c2.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_c2)
    plt.legend(summary2_c2.columns)
    plt.title('American call (ds = 1.0)')
    
    summary2_p2 = pd.DataFrame(output2[1][1]).T.set_index(0, drop =True)
    summary2_p2.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_p2)
    plt.legend(summary2_p2.columns)
    plt.title('American put (ds = 1.0)')   

    #when ds = 1.25
    summary2_c3 = pd.DataFrame(output2[2][0]).T.set_index(0, drop =True)
    summary2_c3.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_c3)
    plt.legend(summary2_c3.columns)
    plt.title('American call (ds = 1.25)')
    
    summary2_p3 = pd.DataFrame(output2[2][1]).T.set_index(0, drop =True)
    summary2_p3.columns = ['EFD', 'IFD', 'CNFD']
    
    plt.figure()
    plt.plot(summary2_p3)
    plt.legend(summary2_p3.columns)
    plt.title('American put (ds = 1.25)')   

    pa2ind = np.where(np.array(summary2_c1.index).round(6) == s0)[0][0]
    ca_2 = summary2_c1.iloc[pa2ind, :]
    pa_2 = summary2_p1.iloc[pa2ind, :]
    
    pb2ind = np.where(np.array(summary2_c2.index).round(6) == s0)[0][0]
    cb_2 = summary2_c2.iloc[pb2ind, :]
    pb_2 = summary2_p2.iloc[pb2ind, :]
    
    pc2ind = np.where(np.array(summary2_c3.index).round(6) == s0)[0][0]
    cc_2 = summary2_c3.iloc[pc2ind, :]
    pc_2 = summary2_p3.iloc[pc2ind, :]

    summary_2call = pd.DataFrame([ca_2, cb_2, cc_2], index = ['ds=0.25', 'ds=1', 'ds=1.25'])
    summary_2put = pd.DataFrame([pa_2, pb_2, pc_2], index = ['ds=0.25', 'ds=1', 'ds=1.25'])
    
    print('Question 2:')
    print('American call price when s0=10:')
    print(summary_2call)
    print('American put price when s0=10:')
    print(summary_2put)
    
