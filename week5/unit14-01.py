#!/usr/bin/env python
# coding: utf-8
# numpy 사용할 때 이 파일 코드 참고하라 

# In[2]:


import csv
f =open('age.csv')
data = csv.reader(f)
next(data)  # 헤더 제거
for row in data :
    print(row)


# In[3]:


import csv
f =open('age.csv')
data = csv.reader(f)
next(data)
home = []  #입력 받은 지역의 데이터를 저장할 리스트 생성
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data :
    if name in row[0] : #입력 받은 지역의 이름이 포함된 행 찾기
        for i in row[3:] : #3번 인덱스 값부터 슬라이싱
            home.append(int(i)) #입력 받은 지역의 데이터를 home에 저장
print(home) #home에 저장된 데이터 출력


# In[6]:


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f)
next(data)
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data :
    if name in row[0] :
        home = np.array(row[3:], dtype = int) # 숫자만 다룰 거니까 numpy 배열로

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)    
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역의 인구 구조')

plt.plot(home) 
plt.show()


# In[8]: numpy로 다루니까 훨씬 효율적이다 (numpy 쓰니까 코드 간결, 빠름)


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f)
next(data)
data = list(data)
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data :
    if name in row[0] :
        home = np.array(row[3:], dtype =int) / int(row[2])  #수정

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)    
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역의 인구 구조')

plt.plot(home) 
plt.show()



# In[10]:


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f) # ➊
next(data)
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data : # ➋
    if name in row[0] :
        home = np.array(row[3:], dtype =int) / int(row[2]) # ➌
for row in data : # ➍
    print(row) # ➎


# In[11]:


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f) # ➊
next(data)
data = list(data)  #추가
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data : # ➋
    if name in row[0] :
        home = np.array(row[3:], dtype =int) / int(row[2]) # ➌
for row in data : # ➍
    print(row) # ➎


# In[12]:


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f) 
next(data)
data = list(data)  #추가
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data : 
    if name in row[0] :
        home = np.array(row[3:], dtype =int) / int(row[2]) 
for row in data : 
    away = np.array(row[3:], dtype = int) / int(row[2])
    print(home - away)


# In[14]:


import numpy as np
import csv
f =open('age.csv')
data = csv.reader(f) 
next(data)
data = list(data)  #추가
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
for row in data : 
    if name in row[0] :
        home = np.array(row[3:], dtype =int) / int(row[2]) 
for row in data : 
    away = np.array(row[3:], dtype = int) / int(row[2])
    print(np.sum(home - away))


# In[ ]:


import numpy as np
import csv
f = open('age.csv')
data = csv.reader(f)
next(data)
data = list(data)
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
mn = 1 # 최솟값을 저장할 변수 생성 및 초기화
result_name = '' # 최솟값을 갖는 지역의 이름을 저장할 변수 생성 및 초기화
result = 0 # 최솟값을 갖는 지역의 연령대별 인구 비율을 저장할 배열 생성 및 초기화
for row in data :
    if name in row[0] :
        home = np.array(row[3:], dtype = int) / int(row[2]) # ➊
    for row in data : # ➏
        away = np.array(row[3:], dtype = int) / int(row[2]) # ➋
        s = np.sum(home - away) # ➌
        if s < mn : # ➍
            mn = s # ➎
            result_name = row[0]
            result = away
            
import matplotlib.pyplot as plt
plt.plot(home)
plt.plot(result)
plt.show()


# In[ ]:


import numpy as np
import csv
#1. 데이터를 읽어온다.
f =open('age.csv')
data = csv.reader(f)
next(data)
data = list(data)
#2. 궁금한 지역의 이름을 입력받는다.
name = input('인구 구조가 알고 싶은 지역의 이름(읍면동 단위)을 입력해주세요 : ')
mn =1
result_name =''
result =0
#3. 궁금한 지역의 인구 구조를 저장한다.
for row in data :
    if name in row[0] :
        home = np.array(row[3:], dtype =int) /int(row[2])
#4. 궁금한 지역의 인구 구조와 가장 비슷한 인구 구조를 가진 지역을 찾는다.
for row in data :
    away = np.array(row[3:], dtype =int) /int(row[2])
    s = np.sum((home - away) **2)
    if s < mn and name not in row[0] :
        mn = s
        result_name = row[0]
        result = away
#5. 궁금한 지역의 인구 구조와 가장 비슷한 곳의 인구 구조를 시각화한다.
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=300)            
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(home, label = name)
plt.plot(result, label = result_name)
plt.legend()
plt.show()


# In[ ]:




