# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:46:29 2025

@author: gusdk
"""

# 저장된 중국_방한외래관광객_2019_202412.csv 파일 불러오기 
import pandas as pd

df = pd.read_csv('중국_방한외래관광객_2019_202412.csv', encoding='cp949')
df.head()

# '입국자 수'에 대한 막대그래프 그리기
import matplotlib.pyplot as plt

plt.figure(figsize=(18,6))
plt.bar(df['입국연월'].astype(str), df['입국자 수'])
plt.title('Number of arrivals to China (monthly)')
plt.xlabel('yyyymm')
plt.ylabel('persons')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False