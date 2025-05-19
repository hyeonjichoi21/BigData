#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 2019년1월 지하철 유임/무임 승하차
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
for row in data :
    print(row)


# In[3]:


# 2019년1월 지하철 유임/무임 승하차, 숫자데이터 정수 변환
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)

for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    print(row)


# In[1]:


# 2019년1월 지하철 유임/무임 승하차, 유임/무임 승하자 비율구하기, zero divide
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    rate = row[4] / row[6] # 0으로 있는 값이 있어서 나누었을 때, 오류 발생
    if rate > mx :
        mx = rate
print(mx)


# In[4]:


# 2019년1월 지하철 유임/무임 승하차, 무임승차가 없는 역 이름
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    if row[6] == 0 :
        print(row)


# In[5]:


# 2019년1월 지하철 유임/무임 승하차, 무임승차 비율이 높은역 찾아 나가기
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    if row[6] != 0 :
        rate = row[4] / row[6]
        if rate > mx :
            mx = rate
            print(row, round(rate,2))


# In[3]:


# 2019년1월 지하철 유임/무임 승하차, 유임승차 비율, 100000만 이상 승차역
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    if row[6] !=0 and (row[4] + row[6]) >100000 : # 무임/유임이 10만보다 높은 게 의미있다고 판단.
        rate = row[4] / (row[4] + row[6]) # 전체 중에 유임승차 비율
        if rate > mx :
            mx = rate
            maxrow=row; maxrate=rate
print(maxrow, round(maxrate,2))


# In[7]:


# 2019년1월 지하철 유임/무임 승하차, 유임승차 비율 94% 이상, 100000만 이상 승차역
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    if row[6] !=0 and (row[4] + row[6]) >100000 :
        rate = row[4] / (row[4] + row[6])
        if rate > 0.94 :
            mx = rate
            print(row, round(rate,2)) # 결과 생각보다 많이 나옴 ㄷㄷ
            


# In[6]:


# # 2019년1월 지하철 유임/무임 승하차, 유임승차 비율 가장큰 역, 100000만 이상 승차역
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = 0
rate = 0
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
    if row[6] !=0 and (row[4] + row[6]) >100000 :
        rate = row[4] / (row[4] + row[6])
        if rate > mx :
            mx = rate
            mx_station = row[3] + ' ' + row[1] # 역이름 저장

print(mx_station, round(mx * 100,2)) # 결괏값이 좀 정리돼서 나오는 걸 볼 수 O


# In[17]:


# 2019년1월 지하철 유임/무임 승하차, 가장 많은역
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = [0] * 4
mx_station = [''] * 4

for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
        if row[i] > mx[i-4] :
            mx[i-4] = row[i]
            mx_station[i-4] = row[3] + ' ' + row[1]
            
for i in range(4) :
    print(mx_station[i], mx[i])


# In[9]:


# 2019년1월 지하철 유임/무임 승하차, 가장 많은역
import csv
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
mx = [0] * 4
mx_station = [''] * 4
label = ['유임승차','유임하차','무임승차','무임하차']
for row in data :
    for i in range(4,8) :
        row[i] = int(row[i])
        if row[i] > mx[i-4] :
            mx[i-4] = row[i]
            mx_station[i-4] = row[3] + ' ' + row[1]
for i in range(4) :
    print(label[i] + ' : ' + mx_station[i], mx[i])


# In[4]:


# 2019년1월 지하철 유임/무임 승하차, 파이차트로 그리기
import csv
import matplotlib.pyplot as plt
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
label = ['유임승차','유임하차','무임승차','무임하차']
list=['서울역', '시청', '건대입구']
plt.rc('font', family = 'Malgun Gothic')
for row in data :
    if row[3] not in list:
        continue
    for i in range(4,8) :
        row[i] = int(row[i])
    plt.figure(dpi = 100)
    plt.pie(row[4:8])
    plt.title(row[1]+':'+row[3])
    plt.axis('equal')
    plt.show()


# In[3]:


import csv;
import matplotlib.pyplot as plt
f = open('subwayfee.csv')
data = csv.reader(f)
next(data)
label = ['유임승차','유임하차','무임승차','무임하차']
c = ['#14CCC0', '#389993', '#FF1C6A', '#CC14AF']

list=['서울역', '시청', '건대입구']
for row in data :
    if row[3] not in list:
        continue
    for i in range(4,8) :
        row[i] = int(row[i])
    plt.figure(dpi = 100)
    plt.rc('font', family = 'Malgun Gothic')
    plt.title(row[3] + ' ' + row[1])
    plt.pie(row[4:8], labels = label, colors = c, autopct = '%1.f%%')
    plt.axis('equal')
    #plt.savefig(row[3] + ' ' + row[1] + '.png')  # 그림파일 생성
    plt.show()


# In[ ]:




