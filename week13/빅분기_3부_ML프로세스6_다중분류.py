#!/usr/bin/env python
# coding: utf-8

# # 8. 머신러닝 프로세스5: 다중분류
# ## 8-1. 데이터 불러오기 및 데이터셋 분할
# In[1]:
# 분석 데이터 불러오기
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data=pd.read_csv('Fvote.csv', encoding='utf-8')
# In[2]:
# 특성치와 레이블 데이터셋 구분
X=data[data.columns[1:13]]
y=data[['parties']]
# In[3]:
# 훈련 데이터, 테스트 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42 )

# ## 8-2. 기본모델 학습
# In[4]:
# ML 알고리즘 모듈 불러오기 및 학습데이터에 적용(LogisticRegression)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, y_train)
# In[5]:
# 분류 예측 결과(0,1)을 'pred_train'에 저장(할당), score로 정확도(accuracy) 확인
pred_train=model.predict(X_train)
model.score(X_train, y_train)
# In[6]:
# 테스트 데이터에 학습데이터의 모델 적용, 'pred_test'에 저장(할당), score로 정확도(accuracy) 확인
pred_test=model.predict(X_test)
model.score(X_test, y_test)
# In[7]:
# 학습데이터의 혼동행렬 보기(정분류, 오분류 교차표)
from sklearn.metrics import confusion_matrix
confusion_train=confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# In[8]:
# 테스트데이터의 혼동행렬 보기(정분류, 오분류 교차표)
confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)

# ## 8-3. Grid Search
# In[9]:
# 그리드서치를 위한 라이브러리 및 탐색 하이퍼파라미터 설정
from sklearn.model_selection import GridSearchCV
param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# In[10]:
# 그리드서치를 로지스틱 모델에 적용하여 훈련데이터 학습
# 교차검증(cv) 5 설정, 훈련데이터 정확도 결과 제시하기(True)
grid_search=GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)
# In[11]:
# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search.best_score_))
# In[12]:
# 테스트 데이터에 적용(C=10), 정확도 결과
print("Test set Score: {:.3f}".format(grid_search.score(X_test, y_test)))

# ## 8-4. Random Search
# In[13]:
# 랜덤서치를 위한 라이브러리 및 탐색 하이퍼파라미터 설정
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs={'C': randint(low=0.001, high=100)}
# In[17]:
# 랜덤서치를 로지스틱 모델에 적용하여 훈련데이터 학습
# 교차검증(cv) 5 설정, 훈련데이터 정확도 결과 제시하기(True)
random_search=RandomizedSearchCV(LogisticRegression(), 
                                 param_distributions=param_distribs, cv=5,
                                 n_iter=1000, # 랜덤횟수 디폴트=10
                                return_train_score=True)
random_search.fit(X_train, y_train)
# In[18]:
# 정확도가 가장 높은 하이퍼파라미터(C) 및 정확도 제시
print("Best Parameter: {}".format(random_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(random_search.best_score_))
# In[19]:
# 테스트 데이터에 최적 텀색 하이퍼 파라미터 적용 정확도 결과
print("Test set Score: {:.3f}".format(random_search.score(X_test, y_test)))
# In[ ]:




