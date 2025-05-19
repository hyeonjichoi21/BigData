#!/usr/bin/env python
# coding: utf-8

# # 10장. 회귀분석 (2) 자동차 연비 예측 분석
# In[1]:
# 향후 버전 업에 대한 경고 메시지 출력 안하기 
import warnings
warnings.filterwarnings(action='ignore') 

# ## 1) 데이터 수집
# In[4]:
import numpy as np
import pandas as pd 
data_df = pd.read_csv('./DATA/auto-mpg.csv', header=0, engine='python')

# ## 2) 데이터 준비 및 탐색
# In[5]:
print(' 데이터셋 크기 : ', data_df.shape)
data_df.head()

# #### - 분석하지 않을 변수 제외하기
# In[4]:
data_df = data_df.drop(['car_name', 'origin', 'horsepower'], axis=1, inplace=False)

# In[5]:
print(' 데이터세트 크기 : ', data_df.shape)
data_df.head()

# In[6]:
data_df.info()

# ## 3) 분석 모델 구축
# In[7]:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# In[8]:
# X, Y 분할하기
Y = data_df['mpg']
X = data_df.drop(['mpg'], axis=1, inplace=False)

# In[9]:
# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# In[10]:
# 선형회귀분석 : 모델 생성
lr = LinearRegression()

# In[11]:
# 선형회귀분석 : 모델 훈련
lr.fit(X_train, Y_train)

# In[12]:
# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)

# ## 4) 결과 분석 및 시각화
# In[13]:
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

# In[14]:
print('Y 절편 값: ',  np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))

# In[15]:
coef = pd.Series(data=np.round(lr.coef_, 2), index=X.columns)
coef.sort_values(ascending=False)

# ### - 회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
# In[16]:
import matplotlib.pyplot as plt
import seaborn as sns

# In[17]:
fig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=2)

x_features = ['model_year', 'acceleration', 'displacement', 'weight', 'cylinders']
plot_color = ['r', 'b', 'y', 'g', 'r']

for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='mpg', data=data_df, ax=axs[row][col], color=plot_color[i])

# ###   <<<< 연비 예측하기  >>>>
# In[18]:
print("연비를 예측하고 싶은 차의 정보를 입력해주세요.")

cylinders_1 = int(input("cylinders : "))
displacement_1 = int(input("displacement : "))
weight_1 = int(input("weight : "))
acceleration_1 = int(input("acceleration : "))
model_year_1 = int(input("model_year : "))

# In[19]:
mpg_predict = lr.predict([[cylinders_1, displacement_1, weight_1, acceleration_1 , model_year_1]])

# In[20]:

print("이 자동차의 예상 연비(mpg)는 %.2f 입니다." %mpg_predict)

