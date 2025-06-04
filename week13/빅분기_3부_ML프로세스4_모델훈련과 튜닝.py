#!/usr/bin/env python
# coding: utf-8

# # 7. 머신러닝 프로세스4: 모델훈련과 세부튜닝
# ## 7-1. 데이터 불러오기 및 데이터셋 분할
# In[1]:
# 분석 데이터 불러오기
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data=pd.read_csv('Fvote.csv', encoding='utf-8')
# In[2]:
# 특성치와 레이블 데이터셋 구분
X=data[data.columns[1:13]]
y=data[['vote']]
# In[3]:
# 훈련 데이터, 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42 )

# ## 7-2. Grid Search
# In[4]:
# 그리드서치를 위한 라이브러리 및 탐색 하이퍼파라미터 설정
from sklearn.model_selection import GridSearchCV
param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# In[5]:
# LogisticRegression 알고리즘 적용
from sklearn.linear_model import LogisticRegression
# In[6]:
# 그리드서치를 로지스틱 모델에 적용하여 훈련데이터 학습
# 교차검증(cv) 5 설정, 훈련데이터 정확도 결과 제시하기(True)
grid_search=GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)
# In[7]:
# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search.best_score_))
# In[8]:
# 테스트 데이터에 적용(C=10), 정확도 결과
print("Test set Score: {:.3f}".format(grid_search.score(X_test, y_test)))
# In[9]:
# 그리드서치 하이퍼파라미터별 상세 결과값
result_grid= pd.DataFrame(grid_search.cv_results_)
result_grid
# In[10]:
# 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(result_grid['param_C'], result_grid['mean_train_score'], label="Train")
plt.plot(result_grid['param_C'], result_grid['mean_test_score'], label="Test")
plt.legend()

# ## 7-3. Random Search
# In[11]:
# 랜덤서치를 위한 라이브러리 및 탐색 하이퍼파라미터 설정
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs={'C': randint(low=0.001, high=100)}
# In[12]:
# LogisticRegression 알고리즘 적용
from sklearn.linear_model import LogisticRegression
# In[13]:
# 랜덤서치를 로지스틱 모델에 적용하여 훈련데이터 학습
# 교차검증(cv) 5 설정, 훈련데이터 정확도 결과 제시하기(True)
random_search=RandomizedSearchCV(LogisticRegression(), 
                                 param_distributions=param_distribs, cv=5,
                                 # n_iter=100, 랜덤횟수 디폴트=10
                                return_train_score=True)
random_search.fit(X_train, y_train)
# In[14]:
# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))
# In[15]:
# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(random_search.score(X_test, y_test)))
# In[16]:
# 랜덤서치 하이퍼파라미터별 상세 결과값
result_random = random_search.cv_results_
pd.DataFrame(result_random)
# In[17]:
# 하이퍼파리미터(C)값에 따른 훈련데이터와 테스트데이터의 정확도(accuracy) 그래프
import matplotlib.pyplot as plt
plt.plot(result_grid['param_C'], result_grid['mean_train_score'], label="Train")
plt.plot(result_grid['param_C'], result_grid['mean_test_score'], label="Test")
plt.legend()
# In[ ]:




