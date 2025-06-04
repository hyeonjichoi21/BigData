#!/usr/bin/env python
# coding: utf-8

# # 6. 머신러닝 프로세스3: 데이터 정규화

# ## 6-1. 데이터 불러오기 및 확인
# In[1]:
# 분석데이터(선거행동) 불러와서 데이터 확인
import pandas as pd
data=pd.read_csv('Fvote.csv', encoding='utf-8')
data.head()
# In[2]:
data.describe()
# In[3]:
data.hist(figsize=(20,10))

# ## 6-2. 특성(X)과 레이블(y) 나누기
# In[4]:
# 특성 변수와 레이블 변수 나누기
X=data.loc[:, 'gender_female':'score_intention']
y=data[['vote']]
# In[5]:
# 특성 변수와 레이블 변수 행열확인
print(X.shape)
print(y.shape)

# ## 6-3. train-test 데이터셋 나누기
# In[6]:
# 학습용 데이터(train)와 테스트용 데이터(test) 구분을 위한 라이브러리 불러오기
# 레이블이 범주형일 경우 straity 옵션 추천
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42)
# In[7]:
# 학습데이터와 테스트데이터의 0/1 비율이 유사한지 평균으로 확인(stratity 옵션 적용시 유사)
print(y_train.mean())
print(y_test.mean())

# ## 6-4. 연속형 특성의 정규화
# ### 가. Min-Max 정규화
# In[8]:
# 특성치(X)의 단위 정규화를 위한 라이브러리 블러오기(min-max)
from sklearn.preprocessing import MinMaxScaler
scaler_minmax=MinMaxScaler()
# In[9]:
# min-max 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 최대-최소 기준이 적용됨
scaler_minmax.fit(X_train)
X_scaled_minmax_train=scaler_minmax.transform(X_train)
# In[10]:
# min-max 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_minmax_train).describe()
# In[11]:
# test 데이터에도 정규화 적용 및 데이터 확인: min-max 방법
X_scaled_minmax_test=scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_minmax_test).describe()
# ### 나. Standardization 정규화
# In[12]:
# 특성치(X)의 단위 정규화를 위한 라이브러리 블러오기(standard)
from sklearn.preprocessing import StandardScaler
scaler_standard=StandardScaler()
# In[13]:
# standard 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 표준화(평균, 표준편차) 기준이 적용됨
scaler_standard.fit(X_train)
X_scaled_standard_train=scaler_standard.transform(X_train)
# In[14]:
# standard 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_standard_train).describe()
# In[15]:
# test 데이터에도 정규화 적용 및 데이터 확인: standard 방법
X_scaled_standard_test=scaler_standard.transform(X_test)
pd.DataFrame(X_scaled_standard_test).describe()

# ## 6-5. 모델 학습
# In[16]:
# ML 알고리즘 모듈 불러오기 및 학습데이터에 적용(LogisticRegression)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
# ### 가. Min-Max 정규화 데이터 적용결과
# In[17]:
# 훈련데이터의 정확도(accuracy) 확인
model.fit(X_scaled_minmax_train, y_train)
pred_train=model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# In[18]:
# 테스트 데이터의 정확도
pred_test=model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# In[19]:
# 학습데이터의 혼동행렬 보기(정분류, 오분류 교차표)
from sklearn.metrics import confusion_matrix
confusion_train=confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# In[20]:
# 테스트데이터의 혼동행렬 보기(정분류, 오분류 교차표)
confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# ### 나. Standardize 정규화 데이터 적용결과
# In[21]:
# 훈련데이터의 정확도(accuracy) 확인
model.fit(X_scaled_standard_train, y_train)
pred_train=model.predict(X_scaled_standard_train)
model.score(X_scaled_standard_train, y_train)
# In[22]:
# 테스트 데이터의 정확도
pred_test=model.predict(X_scaled_standard_test)
model.score(X_scaled_standard_test, y_test)
# In[23]:
# 학습데이터의 혼동행렬 보기(정분류, 오분류 교차표)
from sklearn.metrics import confusion_matrix
confusion_train=confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# In[24]:
# 테스트데이터의 혼동행렬 보기(정분류, 오분류 교차표)
confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)

