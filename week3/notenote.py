# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:54:36 2025

@author: gusdk
"""

import numpy as np
import pandas as pd

df = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])

df.groupby('중간고사').중간고사.count()
df.groupby('중간고사').중간고사.min()
df.groupby(['중간고사']).중간고사.agg([len, min, max]) # 3개 다 구하고 싶으면
df.sort_values(by='중간고사') # 오름차순
df.sort_values(by='중간고사', ascending=False) # 내림차순
df.sort_index(ascending=False) # 인덱스 내림차순
