# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:21:29 2018

@author: SP
"""

class mergeSort:
    
    def divide(self, lst):
        length = len(lst)
        
        if length == 1:
            return lst
        elif length == 2:
            return self.__sort_and_merge(lst[:1],lst[1:])
        else:
            if length%2 == 0:
                mid = int(length/2)
            else:
                mid = int((length+1)/2)            
            left = self.divide(lst[:mid])
            right = self.divide(lst[mid:])
    
            return self.__sort_and_merge(left, right)
        
    
    def __sort_and_merge(self, leftlist, rightlist):
            
        leftlen = len(leftlist)
        rightlen = len(rightlist)
        
        sortedlist = []
        i = 0
        j = 0
        while (i < leftlen) & (j < rightlen):
            if leftlist[i] < rightlist[j]:
                sortedlist.append(leftlist[i])
                i = i+1
            else:
                sortedlist.append(rightlist[j])
                j = j+1
        
        if i < leftlen:
            return sortedlist+leftlist[i:]
        elif j < rightlen:
            return sortedlist+rightlist[j:]
