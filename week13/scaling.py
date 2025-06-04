# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:07:35 2023
@author: Park
"""
###############################################################################

from sklearn.preprocessing import StandardScaler
#data = [[0, 0], [0, 0], [1, 1], [1, 1]]
data = [[1, 0], [2, 0], [3, 0], [4, 1], [5, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.var_)
# [0.5 0.5]
print(scaler.transform(data))
print(scaler.transform([[1.5, 0.5]]))

import math
print( (1.5-3)/math.sqrt(2) ) # z-score
import scipy.stats as ss
ss.zscore([1,2,3,4,5])

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
mu = 0
variance = 1
sigma = math.sqrt(variance)
print(mu+sigma)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()


###############################################################################
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
# MinMaxScaler()
print(scaler.data_max_)
#[ 1. 18.]
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
# [[1.5 0. ]]