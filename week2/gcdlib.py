# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:50:02 2021

@author: Park
"""
def gcd(a, b):
    if a<b:
        a,b=b,a
    if a%b==0:
        return b
    return gcd(b, a%b)

if __name__ == '__main__':
    x,y=[int(n) for n in input('두수 입력:').split()[:2]]
    g=gcd(x,y)
    print("%d, %d 최대공약수= %d" %(x,y,g))
#print(__name__)

import sys
sys.path
