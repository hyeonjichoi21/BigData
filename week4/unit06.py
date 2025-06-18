#!/usr/bin/env python
# coding: utf-8

# In[3]:


# plot 그래프
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(figsize = (10,2), dpi = 300)
plt.plot(result, 'r')
plt.show()


# In[2]:


# histogram
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist([1,1,2,3,4,5,6,6,7,8,10, 5, 3, 4, 2, 2, 2, 2]) # 각 숫자의 빈도...(plot이 아니라 hist)
plt.show() # 빈도 수 나옴 


# In[5]:


# random number
import random 
print('*randint : ', random.randint(1,6)) # 1-6 중 난수 발생

for i in range(5) :
    print(random.randint(1,6))
    
dice = []
for i in range(5) :
    dice.append(random.randint(1,6))
print('*dice : ', dice)


# In[22]:


# 주사위 히스토그램
import random
dice = []
for i in range(5) :
    dice.append(random.randint(1,6))
print(dice)

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[21]:


import random
dice = []
for i in range(100) : # 100번 던지면 6번 던졌을 때보다 확률이 엇비슷해짐
    dice.append(random.randint(1,6))
#print(dice)

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[9]:


# 10000번 주사위를 던지면... 
import random
dice = []
for i in range(1000000) : # 거의 박스형태와 비슷해짐
    dice.append(random.randint(1,6))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(dice, bins = 6)
plt.show()


# In[24]:


# 서울의 최고기온 온도값을 histogram으로 그려본다.
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
plt.figure(dpi = 300)
plt.hist(result, bins = 100, color = 'r')
plt.show()


# In[27]:


plt.figure(dpi = 300)
plt.hist(result, bins = 1000, color = 'r')
plt.show()


# In[28]:


import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
print(len(aug)) # 8월 전체 최고기온
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(aug, bins = 100, color = 'r')
plt.show()


# In[13]: 1월과 8월의 최고 기온 데이터 히스토그램으로 표현하기 


import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08': # 8월
            aug.append(float(row[-1]))
        if month == '01': # 1월 
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.hist(aug, bins = 100, color = 'r', label = 'Aug')
plt.hist(jan, bins = 100, color = 'b', label = 'Jan')
plt.legend()
plt.show()


# In[29]:


# boxplot
import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
import random
result = []
for i in range(13) :
    result.append(random.randint(1,1000))
print(sorted(result))
#import numpy as np
#result = np.array(result)
#print("1/4: " + str(np.percentile(result,25)))
#print("2/4: " + str(np.percentile(result,50)))
#print("3/4: " + str(np.percentile(result,75)))
# result=[1,2,3,4,5] # 등간격이라 일정한 데이터값 추가해도 변화X

# result=[1,2,3,4,5,6, 10] # 등간격 벗어나서 변함. 중간값 4, 1/4, 2/4, 3/4 값이 각각 어디인지도 살펴보기 
plt.boxplot(result) # 박스플롯 - 이상값 찾는 데 유용
plt.show()


# In[15]:


# boxplot - 최고기온
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
result = []

for row in data :
    if row[-1] != '' :
        result.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot(result)
plt.show() # 실행해 보니, 고르지 않고 아주 낮은 값이 들어있구나.


# In[16]:


# 8월, 1월 최고기온 boxplot
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot(aug)
plt.boxplot(jan)
plt.show()


# In[17]: 1월과 8월의 최고 기온 데이터 상자 그림으로 표현하기 


# 8월, 1월 최고기온 boxplot
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
aug = []
jan = []

for row in data :
    month = row[0].split('-')[1]
    if row[-1] != '' :
        if month == '08':
            aug.append(float(row[-1]))
        if month == '01':
            jan.append(float(row[-1]))

import matplotlib.pyplot as plt
#plt.figure(dpi = 300)
plt.boxplot([aug,jan])
plt.show()


# In[18]:


# 12개월 최고기온 boxplot
import matplotlib.pyplot as plt
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)
month = [[],[],[],[],[],[],[],[],[],[],[],[]]

for row in data :
    if row[-1] != '' :
        month[int(row[0].split('-')[1])-1].append(float(row[-1]))

#plt.figure(figsize=(10,5), dpi=300)
plt.boxplot(month)
plt.show()


# In[19]:


# 8월 최고기온 boxplot
import matplotlib.pyplot as plt
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)

day = []
for i in range(31) : 
    day.append([])

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '08':
            day[int(row[0].split('-')[2])-1].append(float(row[-1]))

plt.boxplot(day, showfliers=False)
plt.show()


# In[20]:


# 8월 최고기온 boxplot - 그래프 크기 변경
import matplotlib.pyplot as plt
import csv
f = open('seoul.csv')
data = csv.reader(f)
next(data)

day = [[] for i in range(31)]

for row in data :
    if row[-1] != '' :
        if row[0].split('-')[1] == '08':
            day[int(row[0].split('-')[2])-1].append(float(row[-1]))
        
plt.style.use('ggplot')
plt.figure(figsize=(10,5), dpi=300)
plt.boxplot(day, showfliers=False)
plt.show()


# In[ ]:




