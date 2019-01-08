# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 07:53:02 2019

@author: Serena
"""

class b_search:
    
    def __init__(self, lst, value):
        self.lst = lst.copy()
        self.value = value
    
    def find(self):
        index = list(range(0, len(self.lst)))
        indexlst = list(zip(index, self.lst))
        
        return self.__divide(indexlst)
    
    def __divide(self, indexlst):
        length = len(indexlst)
        
        if length == 0:
            return 0
        elif length == 1:
            if indexlst[0][1] == self.value:
                return indexlst[0][0]
            else:
                return 0
        else:
            slst = sorted(indexlst, key = lambda x: x[1])
            
            mid = round(length/2)
            
            if slst[mid][1] > self.value:
                return self.__divide(slst[:mid])
            elif slst[mid][1] < self.value:
                return self.__divide(slst[mid:])
            else:
                output = []
                j = mid-1
                while slst[j][1] == self.value:
                    output.append(slst[j][0])
                    j-=1
                    if j < 0:
                        break
                    
                output.append(slst[mid][0])
                
                for i in slst[mid+1:]:
                    if i[1] == self.value:
                        output.append(i[0])
                    else:
                        break

                if len(output) == 1:
                    return output[0]
                else:
                    return output
    
##test cases:
import random
random.seed(7)
test = random.sample(range(-10000,10000), k = 2345)
test2 = [108] + test + [108]
b_search(test2, 108).find() #41
