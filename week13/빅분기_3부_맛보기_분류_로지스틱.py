#!/usr/bin/env python
# coding: utf-8

# # 2. 머신러닝 맛보기1: 분류문제
# ## 2-1. 분석 데이터 검토
# In[1]:
# 분석데이터(유방암) 불러와서 데이터 확인
import pandas as pd
data=pd.read_csv('breast-cancer-wisconsin.csv', encoding='utf-8')
data.head()
# In[2]:
# 레이블 변수(유방암) 비율 확인
data['Class'].value_counts(sort=False)
# In[3]:
# 행(케이스수)과 열(컬럼수) 구조 확인
print(data.shape)

# ## 2-2. 특성(X)과 레이블(y) 나누기
# In[4]:
# 특성과 레이블 데이터 나누기: 특성치 데이터셋을 나누는 방법은 다양함
# 방법1: 특성이름으로 특성 데이터셋(X) 나누기
X1=data[['Clump_Thickness', 'Cell_Size', 'Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]
# 방법2: 특성 위치값으로 특성 데이터셋(X) 나누기
X2=data[data.columns[1:10]]
# 방법3: loc 함수로 특성 데이터셋(X) 나누기 (단, 불러올 특성이 연달아 있어야 함)
X3=data.loc[:, 'Clump_Thickness':'Mitoses']
# In[5]:
# 3가지 방법 모두 동일한 특성치 데이터셋 나눠진 결과 확인
print(X1.shape)
print(X2.shape)
print(X3.shape)
# In[6]:
# 레이블 데이터셋 나누기
y=data[["Class"]]
# In[7]:
# 레이블 데이터셋 행, 열 확인
print(y.shape)

# ## 2-3. train-test 데이터셋 나누기
# In[8]:
# 학습용 데이터(train)와 테스트용 데이터(test) 구분을 위한 라이브러리 불러오기
# 레이블이 범주형일 경우 straity 옵션 추천
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X1, y, stratify=y, random_state=42)
# In[9]:
# 학습데이터와 테스트데이터의 0/1 비율이 유사한지 평균으로 확인(stratity 옵션 적용시 유사)
print(y_train.mean())
print(y_test.mean())

# ## 2-4. 정규화
# In[10]:
# 특성치(X)의 단위 정규화를 위한 라이브러리 블러오기(min-max, standard 2가지 비교)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler_minmax=MinMaxScaler()
scaler_standard=StandardScaler()
# ### 가. train data의 정규화
# In[11]:
# min-max 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 최대-최소 기준이 적용됨
scaler_minmax.fit(X_train)
X_scaled_minmax_train=scaler_minmax.transform(X_train)
# In[12]:
# standard 방법으로 정규화
# 주의!: fit은 학습데이터로 해야, 나중에 test 데이터 정규화시 train 데이터의 표준화(평균, 표준편차) 기준이 적용됨
scaler_standard.fit(X_train)
X_scaled_standard_train=scaler_standard.transform(X_train)
# In[13]:
# min-max 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_minmax_train).describe()
# In[14]:
# standard 방법으로 정규화한 데이터의 기술통계량 확인
pd.DataFrame(X_scaled_standard_train).describe()
# ### 나. test data의 정규화
# In[15]:
# test 데이터에도 정규화 적용 및 데이터 확인: min-max 방법
X_scaled_minmax_test=scaler_minmax.transform(X_test)
pd.DataFrame(X_scaled_minmax_test).describe()
# In[16]:
# test 데이터에도 정규화 적용 및 데이터 확인: standard 방법
X_scaled_standard_test=scaler_standard.transform(X_test)
pd.DataFrame(X_scaled_standard_test).describe()

# ## 2-5. 모델 학습
# In[17]:
# ML 알고리즘 모듈 불러오기 및 학습데이터에 적용(LogisticRegression)
# 여기서는 min-max 정규화 데이터로 분석
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_scaled_minmax_train, y_train)
# In[18]:
# 분류 예측 결과(0,1)을 'pred_train'에 저장(할당), score로 정확도(accuracy) 확인
pred_train=model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)
# In[19]:
# 테스트 데이터에 학습데이터의 모델 적용, 'pred_test'에 저장(할당), score로 정확도(accuracy) 확인
pred_test=model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)
# In[20]:
# 학습데이터의 혼동행렬 보기(정분류, 오분류 교차표)
from sklearn.metrics import confusion_matrix
confusion_train=confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)
# In[21]:
# 테스트데이터의 혼동행렬 보기(정분류, 오분류 교차표)
confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)
# In[22]:
# 훈련 데이터의 평가지표 상세 확인
from sklearn.metrics import classification_report
cfreport_train=classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)
# In[23]:
# 테스트 데이터의 평가지표 상세 확인
from sklearn.metrics import classification_report
cfreport_test=classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)
# In[24]:
# ROC 지표 산출을 위한 라이브러리 및 산식
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc = metrics.roc_auc_score(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc
# In[25]:
# ROC Curve 그리기
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='lower right')
plt.show()

# ## 2-6. 예측값 병합 및 저장
# In[26]:
# 학습데이터의 예측범주, 예측확률 컬럼을 생성하여 'y_train' 데이터셋에 추가
prob_train=model.predict_proba(X_scaled_minmax_train)
y_train[['y_pred']]=pred_train
y_train[['y_prob0', 'y_prob1']]=prob_train
y_train
# In[27]:
# 테스트 데이터의 예측범주, 예측확률 컬럼을 생성하여 'y_test' 데이터셋에 추가
prob_test=model.predict_proba(X_scaled_minmax_test)
y_test[['y_pred']]=pred_test
y_test[['y_prob0', 'y_prob1']]=prob_test
y_test
# In[28]:
# 테스트 데이터의 특성치(X_test)와 레이블 및 예측치(y_test)를 병합
Total_test=pd.concat([X_test, y_test], axis=1)
Total_test
# In[29]:
# csv파일로 내보내기 및 저장
Total_test.to_csv("classfication_test.csv")

