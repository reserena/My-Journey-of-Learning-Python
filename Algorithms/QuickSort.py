# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:06:56 2018

@author: Serena
"""

##For use: quickSort(x).qSort(0, len(x)-1)

class quickSort():
    def __init__(self, x, pivot = "Last"):
        self.x = x
        self.pivot = pivot #default: take the last value
        
    def __partition(self, start, end):
           
        if self.pivot == "Last": 
            ##always take the last value as pivot
            p = self.x[end]
            pind = end
            i = int(start-1)
                            
            i = self.__swap(i = i, p=p, pind = pind, start = start, end = end)
                
            self.x[i+1], self.x[pind] = self.x[pind], self.x[i+1]
            return int(i+1)
        elif self.pivot == "First":
            ##always take the first value as pivot
            p = self.x[start]
            pind = start
            i = start                            
            
            i = self.__swap(i = i, p=p, pind = pind, start = start, end = end)
            
            self.x[i], self.x[pind] = self.x[pind], self.x[i]
            return int(i)
        else:
            print("Invalid pivot")
            return None
        
    def __swap(self, i, p, pind, start, end):
        
        for j in range(start, end+1):
            if j == pind:
                continue
            
            if self.x[j] <= p:
                i += 1
                self.x[i], self.x[j] = self.x[j], self.x[i]
        
        return i
        
    def qSort(self, low, high):
        
        if (low < high):
            p = self.__partition(start = low, end = high)
            
            self.qSort(low = low, high = p-1)
            self.qSort(low = p+1, high = high)
