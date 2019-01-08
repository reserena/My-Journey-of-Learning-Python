# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:04:59 2018

@author: Serena
"""
##Using the same algorithm as MergeSort
##Example:
##Input: [5,1,3,6,2,4]
##Output: 7

class countInv:
    
    #counting function added on 1/8
    def counting(self, original):
        
        a = original.copy()
        
        return self.__divide(a)[1]
    
    def __divide(self, a):
        lofa = len(a)
        if lofa == 1:
            return a, 0
        else:

            mid = round(lofa/2)
            left, x = self.__divide(a[:mid])
            right, y = self.__divide(a[mid:])
            count = x + y
            
            return self.__merge_and_count(left, right, count)
        
    def __merge_and_count(self, b, c, cinv):
        
        blen = len(b)
        clen = len(c)
        
        i = 0
        j = 0
        output = []
        while (i < blen) and (j < clen):
            if b[i] <= c[j]:
                output.append(b[i])
                i+=1
            else:
                output.append(c[j])
                j+=1
                cinv = cinv+ blen-i
                
        if i < blen:
            return output+b[i:], cinv
        
        if j < clen:
            return output+c[j:], cinv
                
##test
test = [5,1,3,6,2,4]
countInv().counting(test)
