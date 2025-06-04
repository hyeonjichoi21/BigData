#!/usr/bin/env python
# coding: utf-8

# # 6. 데이터정제 실전과제

# ## 6-1. 데이터 불러오기 및 탐색
# In[1]:
import pandas as pd
data=pd.read_csv('house_raw.csv')
data.head()
# In[2]:
data.describe()
# In[3]:
data.hist(bins=50, figsize=(20,15))

# ## 6-2. 선형회귀 적용 (정제 전 데이터)
# In[4]:
# 특성데이터셋, 레이블 데이터셋 나누기
X=data[data.columns[0:5]]
y=data[["house_value"]]
# In[5]:
# 학습용 데이터(train)와 테스트용 데이터(test) 구분을 위한 라이브러리 불러오기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)
# In[11]:
# 데이터 정규화(min-max)를 위한 라이브러리 설정
from sklearn.preprocessing import MinMaxScaler
scaler_minmax=MinMaxScaler()

# 훈련데이터 및 데스트데이터 정규화
scaler_minmax.fit(X_train)
X_scaled_minmax_train=scaler_minmax.transform(X_train)
X_scaled_minmax_test=scaler_minmax.transform(X_test)
df = pd.DataFrame(X_scaled_minmax_train, columns = data.columns[0:5])
df.hist(bins=50, figsize=(20,15))
# In[12]:
# 선형 모델 적용
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_scaled_minmax_train, y_train)
# In[13]:
# 훈련데이터의 정확도(R-square: 설명력) 확인
pred_train=model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train) # 
# In[14]:
# 테스트데이터의 정확도(R-square: 설명력) 확인
pred_test=model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test) 

# ## 6-3. 데이터 정제를 위한 세부 검토
# ### 가. bedrooms
# In[15]:
# bedrooms 변수의 상세 분포 확인
data_bedroom=data[data['bedrooms']<0.6]
data_bedroom['bedrooms'].hist(bins=100, figsize=(20,15))
# In[16]:
# bedrooms 변수의 이상치 데이터 확인
data_bedroom2=data[data['bedrooms']>=0.6]
print(data_bedroom2['bedrooms'].value_counts())
data_bedroom2
# ### 나. households
# In[17]:
# households 변수의 상세 분포 확인
data_households=data[data['households']<10]
data_households['households'].hist(bins=100, figsize=(20,15))
# In[19]:
# households 변수의 이상치 데이터 확인
data_households2=data[data['households']>=10]
print(data_households2['households'].value_counts())
data_households2
# ### 다. rooms
# In[18]:
# rooms 변수의 상세 분포 확인
data_room=data[data['rooms']<20]
data_room['rooms'].hist(bins=100, figsize=(20,15))
# In[21]:
# bedrooms 변수의 이상치 데이터 확인
data_room2=data[data['rooms']>=20]
print(data_room2['rooms'].value_counts())
data_room2

# ## 6-4. 정제 데이터셋 생성
# In[19]:
# 정상데이터셋(new_data) = 침실 0.5미만, 가족수 7명 미만, 방 12개 미만인 데이터
new_data=data[(data['bedrooms']<0.5) & (data['households']<7) & (data['rooms']<12)]
# In[20]:
new_data.describe()
# In[21]:
new_data.hist(bins=50, figsize=(20,15))

# ## 6-5. 선형회귀 적용 (정제 후 데이터)
# In[22]:
# 특성데이터셋, 레이블 데이터셋 나누기
X=new_data[new_data.columns[0:5]]
y=new_data[["house_value"]]

# 학습용 데이터(train)와 테스트용 데이터(test) 구분을 위한 라이브러리 불러오기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)

# 데이터 정규화(min-max)
from sklearn.preprocessing import MinMaxScaler
scaler_minmax=MinMaxScaler()

# 훈련데이터 및 데스트데이터 정규화
scaler_minmax.fit(X_train)
X_scaled_minmax_train=scaler_minmax.transform(X_train)
X_scaled_minmax_test=scaler_minmax.transform(X_test)

# 선형 모델 적용
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

# 훈련데이터의 정확도(R-square: 설명력) 확인
pred_train=model.predict(X_scaled_minmax_train)
print("훈련데이터 정확도", model.score(X_scaled_minmax_train, y_train))

# 테스트데이터의 정확도(R-square: 설명력) 확인
pred_test=model.predict(X_scaled_minmax_test)
print("테스트데이터 정확도", model.score(X_scaled_minmax_test, y_test))

# In[23]:
# 최종 데이터 저장
new_data.to_csv('house_price.csv', index=False)

# In[ ]:




