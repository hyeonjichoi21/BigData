# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:50:02 2021

@author: Park
"""
import gcdlib

def lcm(a, b):
    g=gcdlib.gcd(a,b)
    x,y=a//g, b//g
    return x*y*g

if __name__ == '__main__':
    x,y=[int(n) for n in input('두수 입력:').split()[:2]]
    l=lcm(x,y)
    print("%d, %d 최소공배수= %d" %(x,y,l))
#print(__name__)