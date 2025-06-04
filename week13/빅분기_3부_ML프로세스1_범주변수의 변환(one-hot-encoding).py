#!/usr/bin/env python
# coding: utf-8

# # 4. 머신러닝 프로세스1: 범주형 특성변수의 원핫인코딩 
# - (One Hot Encoding)

# ## 4-1. 데이터 범주-연속-레이블로 나누기
# In[1]:
# vote(유권자 선거행동) 데이터셋 불러오기 및 확인
import pandas as pd
data=pd.read_csv('vote.csv', encoding='utf-8')
data.head()
# In[3]:
# 범주변수와 기타 변수를 각각 X1과 XY로 나누기
X1=data[['gender', 'region']]
XY=data[['edu', 'income', 'age', 'score_gov', 'score_progress', 'score_intention', 'vote', 'parties']]

# ## 4-2. 범주형 변수의 One-hot-encoding 변환
# In[4]:
# 성별(gender)과 출신지역(region)의 숫자를 문자로 변환
X1['gender'] = X1['gender'].replace([1,2], ['male', 'female'])
X1['region'] = X1['region'].replace([1,2,3,4,5], ['Sudo', 'Chungcheung', 'Honam', 'Youngnam', 'Others'])
# In[5]:
# 변환된 범주형 데이터(X1) 확인
X1.head()
# In[6]:
# 범주변수를 one-hot-encoding으로 변환 및 확인
X1_dum=pd.get_dummies(X1)
X1_dum.head()

# ## 4-3. 자료 통합 및 저장하기
# In[7]:
# 변환 데이터와 기타 변수를 한 데이터셋으로 통합 및 확인
Fvote=pd.concat([X1_dum, XY], axis=1)
Fvote.head()
# In[8]:
# 통합된 데이터를 'Fvote.csv' 파일로 저장 내보내기
Fvote.to_csv('Fvote.csv', index=False, sep=',', encoding='utf-8')

# In[ ]:




