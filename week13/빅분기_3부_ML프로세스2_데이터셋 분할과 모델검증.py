#!/usr/bin/env python
# coding: utf-8

# # 5. 머신러닝 프로세스2: 데이터셋 분할과 모델검증
# - 훈련데이터 및 테스트 데이터셋 분할
# - 홀드아웃과 교차검증

# ## 5-1. 특성치(X), 레이블(y) 나누기
# In[1]:
# 데이터셋 불어오기 및 확인
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data=pd.read_csv('Fvote.csv', encoding='utf-8')
data.head()
# In[2]:
# 특성변수 데이터셋 나누기 
# 방법1: 특성이름으로 데이터셋 나누기 
X=data[['gender_female', 'gender_male', 'region_Chungcheung', 'region_Honam', 
        'region_Others', 'region_Sudo', 'region_Youngnam', 'edu', 'income', 
        'age', 'score_gov', 'score_progress', 'score_intention']]
# 방법2: 특성 위치값으로 데이터셋 나누기 
X=data[data.columns[1:14]]
# 방법3: loc 함수로 데이터셋 나누기 (단, 불러올 특성이 연달아 있어야 함)
X=data.loc[:, 'gender_female':'score_intention']
# In[3]:
# 레이블 변수 중 투표여부(vote) 데이터셋 나누기
y=data[["vote"]]

# ## 5-2. train-test 데이터셋 나누기
# In[4]:
# 훈련데이터와 테스트 데이터 셋 나누기 및 데이터 확인
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42)
# In[5]:
# 데이터셋 행렬 구조 확인
print(X_train.shape)
print(X_test.shape)

# ## 5-3. 모델 적용
# In[6]:
# LogisticRegression 알고리즘 적용
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
# ### 가. 랜덤없는 교차검증: cross_val_score
# In[7]:
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, X_train, y_train, cv=5)
print("5개 테스트 셋 정확도:", scores)
print("정확도 평균:", scores.mean())
# ### 나. 랜덤 있는 교차검증: K-Fold
# In[8]:
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5, shuffle=True, random_state=42)
score=cross_val_score(model, X_train, y_train, cv=kfold)
print("5개 폴드의 정확도:", scores)
# ### 다. 임의분할 교차검증
# In[8]:
from sklearn.model_selection import ShuffleSplit
shuffle_split=ShuffleSplit(test_size=0.5, train_size=0.5, random_state=42)
score=cross_val_score(model, X_train, y_train, cv=shuffle_split)
print("교차검증 정확도:", scores)

# ## 5-4. train-validity-test 분할과 교차검증
# In[9]:
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test=train_test_split(X, y, random_state=1 )
X_train, X_valid, y_train, y_valid=train_test_split(X_train_val, y_train_val, random_state=2 )
# In[10]:
model.fit(X_train, y_train)
scores=cross_val_score(model, X_train, y_train, cv=5)
print("교차검증 정확도:", scores)
print("정확도 평균:", scores.mean())
# In[11]:
model.score(X_valid, y_valid)
# In[12]:
model.score(X_test, y_test)

