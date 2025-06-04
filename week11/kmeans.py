# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:10:40 2023

@author: USER
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
X = np.array([[2, 3], [1, 3], [2, 2.5], [8, 3], [10, 3], [4, 4]])
#X = np.array([[2, 3], [9, 3], [6, 3]])
data=pd.DataFrame(X, columns=['x', 'y'])
data.info()
data.plot(kind="scatter", x="x",y="y",figsize=(5,5),color="red")

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_ # array[1,1,1,0,0,1]
# 아래 문장에 오류 있으면 pip install threadpoolctl==3.1.0 후 restart the kernel
kmeans.predict( [[0, 0], [12, 3]] ) # 얘측해 봐라
centers=kmeans.cluster_centers_ # 중심점 잡기 
print(centers)

# cluster center 그려보기
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1], marker='o')
plt.scatter(centers[:,0], centers[:,1], marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
