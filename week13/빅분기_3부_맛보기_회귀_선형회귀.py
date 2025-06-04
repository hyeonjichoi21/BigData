#!/usr/bin/env python
# coding: utf-8

# # 3. 머신러닝 맛보기2: 회귀문제
# ## 3-1. 분석 데이터 검토
# In[1]:
# 분석데이터(주택가격) 불러와서 데이터 확인
import pandas as pd
data=pd.read_csv('house_price.csv', encoding='utf-8')
data.head()
# In[2]:
# 행(케이스수)과 열(컬럼수) 구조 확인
print(data.shape)
# In[3]:
# 변수별 기술통계 확인
data.describe()
# In[4]:
# 변수별 히스토그램 확인
data.hist(bins=50, figsize=(20,15))

# ## 3-2. 특성(X)과 레이블(y) 나누기
# In[5]:
# 방법1: 특성이름으로 특성 데이터셋(X) 나누기
X1=data[['housing_age', 'income', 'bedrooms', 'households', 'rooms']]
# 방법2: 특성 위치값으로 특성 데이터셋(X) 나누기
X2=data[data.columns[0:5]]
# 방법3: loc 함수로 특성 데이터셋(X) 나누기 (단, 불러올 특성이 연달아 있어야 함)
X3=data.loc[:, 'housing_age':'rooms']
# In[6]:
# 3가지 방법 모두 동일한 특성치 데이터셋 나눠진 결과 확인
print(X1.shape)
print(X2.shape)
print(X3.shape)
# In[7]:
# 레이블 데이터셋 나누기
y=data[["house_value"]]
# In[8]:
# 레이블 데이터셋 행, 열 확인
print(y.shape)

# ## 3-3. train-test 데이터셋 나누기
# In[9]:
# 학습용 데이터(train)와 테스트용 데이터(test) 구분을 위한 라이브러리 불러오기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X1, y, random_state=42)
# In[10]:
print(y_train.mean())
print(y_test.mean())

# ## 3-4. 정규화
# In[11]:
# 특성치(X)의 단위 정규화를 위한 라이브러리 블러오기(min-max, standard 2가지 비교)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler_minmax=MinMaxScaler()
scaler_standard=StandardScaler()
# ### 가. train data의 정규화
# In[12]:
# min-max 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 최대-최소 기준이 적용됨
scaler_minmax.fit(X_train)
X_scaled_minmax_train=scaler_minmax.transform(X_train)
# In[13]:
# standard 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 최대-최소 기준이 적용됨
scaler_standard.fit(X_train)
X_scaled_standard_train=scaler_standard.transform(X_train)
# In[14]:
# min-max 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_minmax_train).describe()
# In[15]:
# standard 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_standard_train).describe()
# ### 나. test data의 정규화
# In[16]:
# test 데이터에도 정규화 적용 및 데이터 확인: min-max 방법
X_scaled_minmax_test=scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_minmax_test).describe()
# In[17]:
# test 데이터에도 정규화 적용 및 데이터 확인: standard 방법
X_scaled_standard_test=scaler_standard.transform(X_test)
pd.DataFrame(X_scaled_standard_test).describe()

# ## 3-5. 모델 학습
# In[18]:
# ML 알고리즘 모듈 불러오기 및 학습데이터에 적용(LinearRegression)
# 여기서는 min-max 정규화 데이터로 분석
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_scaled_minmax_train, y_train)
# In[19]:
# 회귀 예측 결과(주택가격)을 'pred_train'에 저장(할당), score로 정확도(R-square: 설명력) 확인
pred_train=model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# In[20]:
# 테스트 데이터에 학습데이터의 모델 적용, 'pred_test'에 저장(할당), score로 정확도(R-square: 설명력) 확인
pred_test=model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# In[21]:
# 기타 선형 모델평가지표: RMSE (Root Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(y_test, pred_test)
np.sqrt(MSE)
# In[22]:
# 기타 선형 모델평가지표: MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred_test)
# In[23]:
# 기타 선형 모델평가지표: MSE (Mean Squared Error)
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test, pred_test)
# In[24]:
# 기타 선형 모델평가지표: MAPE (Mean Absolute Percentage Error)
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - pred_test) / y_test)) * 100 
MAPE(y_test, pred_test)
# In[25]:
# 기타 선형 모델평가지표: MPE (Mean Percentage Error)
def MAE(y_test, y_pred):
    return np.mean((y_test - pred_test) / y_test) * 100
MAE(y_test, pred_test)

# ## 3-6. 예측값 병합 및 저장
# In[26]:
# 학습데이터의 예측값(주택가격) 컬럼을 생성하여 'y_train' 데이터셋에 추가
prob_train=model.predict(X_scaled_minmax_train)
y_train[['y_pred']]=pred_train
y_train
# In[27]:
# 테스트 데이터의 예측값(주택가격) 컬럼을 생성하여 'y_test' 데이터셋에 추가
prob_test=model.predict(X_scaled_minmax_test)
y_test[['y_pred']]=pred_test
y_test
# In[28]:
# 테스트 데이터의 특성치(X_test)와 레이블 및 예측치(y_test)를 병함
Total_test=pd.concat([X_test, y_test], axis=1)
Total_test
# In[29]:
# csv파일로 내보내기 및 저장
Total_test.to_csv("regression_test.csv")

