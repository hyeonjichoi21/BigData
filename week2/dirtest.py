# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:59:02 2022

@author: Park
"""

# doc 
from uchang import test1 
y=test1(10)
print(y)

import uchang
cal1 = uchang.Calculator()
cal2 = uchang.Calculator()

print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))

import uchang
a = uchang.Math()
print(a.solv(2))
result=uchang.add(3,4)

type(test1)
# dir
dir(test1)
dir(test1.__doc__)

import uchang
dir(uchang)
uchang.__doc__
print(uchang.__doc__)
print(uchang.Math.__doc__)

import pandas as pd
dir(pd)
mydf=pd.DataFrame([2,5,7])
dir(mydf) # 유용함...

import mod2
a = mod2.Math()
print(a.solv(2))
result=mod2.add(3,4)

import sklearn
dir(sklearn)
sklearn.__package__
import sklearn.ensemble
dir(sklearn.ensemble)
