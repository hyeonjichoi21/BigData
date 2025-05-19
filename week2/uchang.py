# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 04:23:10 2022
@author: Park

import uchang
a = uchang.Math()
print(a.solv(2))
result=uchang.add(3,4)

"""

def test1(x):
    return x

def test2(y):
    return y

class Calculator:
    def __init__(self):
        self.result = 0
    def add(self, num):
        self.result += num
        return self.result
    
PI = 3.141592


class Math: 
    """
    # a = uchang.Math()
    # print(a.solv(2))
    """
    def solv(self, r): 
        return PI * (r ** 2) 

def add(a, b): 
    return a+b 

"""
Have nice day
"""