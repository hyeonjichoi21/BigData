#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
f = open('seoul.csv')
data = csv.reader(f)
for row in data :
    print(row)


# In[2]:


# next(data)
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
for row in data :
    print(row[-1])


# In[3]:


import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = [] # 최고 기온 데이터를 저장할 리스트 생성

for row in data :
    if row[-1] != '' : # 최고 기온 데이터 값이 존재한다면
        result.append(float(row[-1])) # result 리스트에 최고 기온 값 추가
print(result) # result에 리스트 타입의 데이터들이 들어 있음



# In[4]:


print(len(result))


# In[1]:


import csv
import matplotlib.pyplot as plt
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

plt.plot(result, 'r') # result 리스트에 저장된 값을 빨간색 그래프로 그리기
plt.title('max temperature')
plt.show() # 그래프 나타내기


# In[5]: 그림 크기 조정


import matplotlib.pyplot as plt
# plt.figure(figsize = (10,2))
plt.figure(figsize = (10,2), dpi = 300)
plt.title('max temperature')
plt.plot(result, 'r')
plt.show()


# In[6]:


# 연월일 분리해보기
s = 'hello python'
print(s);  s.split() # <- 문자열들을 공백 기준을 나눔 
date = '1907-10-01'
print(date.split('-')); 
print(date.split('-')[0])
print(date.split('-')[1])
print(date.split('-')[2])


# In[7]:


# 8월의 온도값 조사
import csv

f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' : # 결측치  (null값)
        if row[0].split('-')[1] == '08' :
            result.append(float(row[-1]))

print(len(result)) # 전체 데이터 개수
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.plot(result, 'hotpink')
plt.title('August max temperature')
plt.show()


# In[8]:


# 매년 2월 14일 최고기온 데이터로 그려보기
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '14' :
            result.append(float(row[-1]))
print(len(result)) # 전체 데이터 개수
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.title('Feb 14 max temperature')
plt.plot(result, 'hotpink')
plt.show()


# In[17]:


# 매년 2월 14일 최고기온/최저기온 데이터로 그려보기
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data :
    if row[-1] != '' and row[-2] != '' :
        if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '14' :
            high.append(float(row[-1]))
            low.append(float(row[-2]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.title('August max/min temperature')
plt.plot(high, 'hotpink')
plt.plot(low, 'skyblue')
plt.show()


# In[13]:


# 1983년 이후 2월 14일 최고기온/최저기온 데이터로 그려보기
import csv

f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data :
    if row[-1] != '' and row[-2] != '' :
        if 1983 <= int(row[0].split('-')[0]) :
            if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '14' :
                high.append(float(row[-1]))
                low.append(float(row[-2]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.plot(high, 'hotpink')
plt.plot(low, 'skyblue')
plt.show()


# In[10]:


# 1983년 이후 2월 14일 최고기온/최저기온 데이터로 그려보기
import csv

f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = []
low = []

for row in data :
    if row[-1] != '' and row[-2] != '' :
        if 1983 <= int(row[0].split('-')[0]) :
            if row[0].split('-')[1] == '02' and row[0].split('-')[2] == '14' :
                high.append(float(row[-1]))
                low.append(float(row[-2]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.plot(high, 'hotpink')
plt.plot(low, 'skyblue')

plt.rc('font', family ='Malgun Gothic')       #한글폰트사용하기
plt.title('내 생일의 기온 변화 그래프')
#plt.rcParams['axes.unicode_minus']=False     #한글폰트사용시 마이너스 부호
plt.show()


# In[11]:


# 1983년 이후 2월 14일 최고기온/최저기온 데이터로 그려보기
import csv
import matplotlib.pyplot as plt

f = open('seoul.csv')
data = csv.reader(f)
next(data)
high = [] # 최고 기온 값을 저장할 리스트 high 생성
low = [] # 최저 기온 값을 저장할 리스트 low 생성

for row in data :
    if row[-1] != '' and row[-2] != '' : # 최고 기온 값과 최저 기온 값이 존재한다면
        date = row[0].split('-') # 날짜 값을 – 문자를 기준으로 구분하여 저장
        if 1983 <= int(date[0]) : # 1983년 이후 데이터라면
            if date[1] == '02' and date[2] == '14' : # 2월 14일이라면
                high.append(float(row[-1])) # 최고 기온 값을 high 리스트에 저장
                low.append(float(row[-2])) # 최저 기온 값을 low 리스트에 저장

                plt.rc('font', family = 'Malgun Gothic') # 맑은 고딕을 기본 글꼴로 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.title('내 생일의 기온 변화 그래프') # 제목 설정
plt.plot(high, 'hotpink', label = 'high') # high 리스트에 저장된 값을 hotpink 색으로 그리고 레이블을 표시

plt.plot(low, 'skyblue', label = 'low') # low 리스트에 저장된 값을 skyblue 색으로 그리고 레이블을 표시
plt.legend() # 범례 표시
plt.show() # 그래프 나타내기


# In[ ]:




