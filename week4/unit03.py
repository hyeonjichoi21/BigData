#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
f =open('seoul.csv')
data = csv.reader(f) # !!! csv로 읽으면 모든 데이터 타입이 문자형이다.
header =next(data)
for row in data :
    row[-1] = float(row[-1]) # 최고 기온을 "실수"로 변형, -1은 끝에서 첫번째=row[4]
    #row[4] = float(row[4]) # '최고 기온'이란 열은 실수로 변형 ㅋㅋ 이해 완 
    print(row)
f.close()


# In[1]:


# 최고 기온 값 찾기
import csv
max_temp =-999   # 최고 기온 값을 저장할 변수
max_date =''       # 최고 기온이 가장 높았던 날짜를 저장할 변수
f =open('seoul.csv')
data = csv.reader(f)
header =next(data)
for row in data :
    if row[-1] =='' :
        row[-1] =-999   # -999를 넣어 빈 문자열이 있던 자리라고 표시
    row[-1] = float(row[-1]) 
    if max_temp < row[-1] :
        max_date = row[0]
        max_temp = row[-1]
f.close()
print('기상 관측 이래 서울의 최고 기온은=>',max_date+'로, ', max_temp, '도 였습니다.')


# In[7]:
# dataFrame으로 하면 for문이 필요없다앙

import pandas as pd
datapd = pd.read_csv("seoul.csv", encoding='cp949')
maxtemp=max(datapd['최고기온(℃)'])
print(datapd[datapd['최고기온(℃)'] == datapd['최고기온(℃)'].max()])
index=datapd[datapd['최고기온(℃)'] == datapd['최고기온(℃)'].max()].index
print('기상 관측 이래 서울의 최고 기온은=>',datapd.loc[index]['날짜'].to_string(index=False)+'로, ', max_temp, '도 였습니다.')


# In[ ]:




