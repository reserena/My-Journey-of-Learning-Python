# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:06:56 2018

@author: Serena
"""

##For use: quickSort(x).qSort(0, len(x)-1)

class quickSort():
    '''Pivot option:
    First: always take the first value
    Last: always take the last value
    median: randomized pivot, take the median of the first, middle, and last value'''
    
    def __init__(self, x, pivot = "median", c = 0):
        self.x = x
        self.pivot = pivot #default: take the median
    
    def qSort(self, low, high):
        
        if (low < high):
            
            p = self.__partition(start = low, end = high)
            
            self.qSort(low = low, high = p-1)
            self.qSort(low = p+1, high = high)
            
                        
    def __partition(self, start, end):
        
        import statistics as st
        
        if self.pivot == "last": 
            ##always take the last value as pivot
            p = self.x[end]
            self.x[start], self.x[end] = self.x[end], self.x[start]           
                                        
        elif self.pivot == "first":
            ##always take the first value as pivot
            p = self.x[start]
            

        elif self.pivot == "median":
            first = self.x[start]
            middle = self.x[int((end-start)/2+start)]
            last = self.x[end]
            
            p = st.median([first, middle, last])
            mind = [first, middle, last].index(p)
            pind = [start, int((end-start)/2+start), end][mind]
            
            self.x[pind], self.x[start] = self.x[start], self.x[pind]
            
        else:
            print("Invalid pivot")
            return None
        
        pind = start
                           
        i = self.__swap(i = start, p=p, pind = pind, start = start, end = end)
        
        self.x[i], self.x[pind] = self.x[pind], self.x[i]
        return int(i)
        
    def __swap(self, i, p, pind, start, end):
        
        for j in range(start+1, end+1):
            
            if self.x[j] <= p:
                i += 1
                if i==j:
                    continue
                else:
                    self.x[i], self.x[j] = self.x[j], self.x[i]
        
        return i
            
#test
test = [8,10,2,3,9,7,5,1,6,0,4]
quickSort(test, pivot = "first").qSort(0, len(test)-1) 
