#!/usr/bin/env python
# coding: utf-8

# In[6]:


# seoul.csv 파일은 기상청(https://data.kma.go.kr)에서 받을 수 있음
import csv 
f = open('seoul.csv', 'r', encoding='cp949') 
data = csv.reader(f, delimiter=',') 
for row in data :
    print(row) # 데이터 한 행씩 읽어오기 
f.close() 


# In[3]: next() 함수를 활용해 헤더 저장하기 


import csv
f =open('seoul.csv')
data = csv.reader(f)
header = next(data)  #①
print(header)        #② 헤더만 읽음 
f.close()


# In[4]:


import csv
f =open('seoul.csv')
data = csv.reader(f)
header =next(data) # next()는 데이터 위치를 다음으로 이동...
for row in data : # 헤더 제외하고 출력 
    print(row)
f.close()


# In[2]:


# pandas 를 사용하여 데이터 다루기..... 비교해보자
# -*- coding: euc-kr -*-
import pandas as pd
datapd = pd.read_csv("seoul.csv",  encoding='cp949')
datapd.head()
print(datapd) # 앞 5개 - 뒤 5개 출력됨
print(datapd[0:5]); datapd.info() # 각종 정보 알려줌 



# In[4]:


# pandas 를 사용하여 데이터 다루기..... 비교해보자
# -*- coding: euc-kr -*-
import pandas as pd
datapd = pd.read_csv("seoul.csv",  encoding='ANSI')
datapd.head()
print(datapd)
print(datapd[0:5])


# In[ ]:




