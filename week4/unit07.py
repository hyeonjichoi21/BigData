#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 인구 공공데이터 받기 => [www.mois.go. kr] –[정책자료] –[통계] –[주민등록 인구통계 ]
import csv

f = open('age.csv')
data = csv.reader(f)
for row in data :
    print(row)


# In[3]:


print('신도림' in '서울특별시 구로구 신도림동(1153051000)')
print('1153' in '서울특별시 구로구 신도림동(1153051000)')
print('()' in '서울특별시 구로구 신도림동(1153051000)')


# In[5]:

# 일반 csv 프로그래밍 
import csv
f = open('age.csv')
data = csv.reader(f)

for row in data :
    if '신도림' in row[0] :
        print(row)

# pandas 프로그래밍
import pandas as pd
mydata=pd.read_csv("./age.csv", encoding='cp949')
print( mydata[mydata['행정구역'].isin(["신도림"])])
print(mydata[mydata['행정구역'].str.contains('신도림')].iloc[:,3:]) # 앞에꺼 빼고 값들만 ~...

# In[6]:


import csv
f = open('age.csv')
data = csv.reader(f)

for row in data :
    if '신도림' in row[0] :
        for i in row[3:] :  # 4번째 열부터 모두 출력...
            print(i)


# In[7]:


import csv
f = open('age.csv')
data = csv.reader(f)
result = []
for row in data :
    if '신도림' in row[0] :
        for i in row[3:] :
            result.append(int(i))
print(result)


# In[7]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(result)

plt.show()


# In[8]:
# 중계4동 쳐봄 -> 재밌음 ㅋ-ㅋ

import csv
f = open('age.csv')
data = csv.reader(f)
result = []
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data :
    if name in row[0] :
        print(name+'동이 포함된 지역 :', row[0])
        for i in row[3:] : # 3부터 끝까지 
            result.append(int(i))
            
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('font', family = 'Malgun Gothic')
plt.title(name +' 지역의 인구 구조')
plt.plot(result)
plt.show()


# In[10]:


# 동이름 입력 받기... Uchang
import csv
f = open('age.csv')
data = csv.reader(f)
result = []
dong=input('동 이름 입력...=>')
for row in data :
    if dong in row[0] :
        print(dong+'동이 포함된 지역 :', row[0])
        print(row)        
        
f = open('age.csv')  # 파일 다시 열기
data = csv.reader(f)
result = []
print(result)
for row in data :
    if dong in row[0] :
        print(dong+'동이 포함된 지역 :', row[0])
        for i in row[3:] :
            result.append(int(i))
print(result)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(result)
plt.show()


# In[ ]:




