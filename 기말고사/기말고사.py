# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:02:56 2025

@author: USER
"""

# 코드 실행 상태가 초기화되었으므로 다시 필요한 파일을 복원 후 작업을 반복
import pandas as pd
import numpy as np
# import 문 필요

# 데이터 
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

# 인코딩 - 명목형 변수
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder();
train['age'] = le.fit_transform(train['age'])
test['age'] = le.fit_transform(test['age'])
# 실행하면 X_train의 Warehouse_block의 값이 다 "숫자"로 바뀐 것을 알 수 있다.

train['income'] = le.fit_transform(train['income'])
test['income'] = le.fit_transform(test['income'])


train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

train['region'] = le.fit_transform(train['region'])
test['region'] = le.fit_transform(test['region'])

train['num_logins'] = le.fit_transform(train['num_logins'])
test['num_logins'] = le.fit_transform(test['num_logins'])

train['total_purchase'] = le.fit_transform(train['total_purchase'])
test['total_purchase'] = le.fit_transform(test['total_purchase'])

train['avg_session_time'] = le.fit_transform(train['avg_session_time'])
test['avg_session_time'] = le.fit_transform(test['avg_session_time'])


train['days_since_last_login'] = le.fit_transform(train['days_since_last_login'])
test['adays_since_last_login'] = le.fit_transform(test['days_since_last_login'])

# 스케일링
###
# 파생변수 생성
train['product_interest_new'] = (train['product_interest'] > 0).astype(int)
#test['product_interest_new'] = (test['product_interest'] > 0).astype(int)
train.drop(columns=['product_interest'], inplace=True)
#test.drop(columns=['product_interest'], inplace=True)

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

# 분석에 필요하지 않은 컬럼 제거 (예: 상관관계 낮은 컬럼)
#x_train.drop(columns=['최대구매액'], inplace=True)
#x_test.drop(columns=['최대구매액'], inplace=True)

# train-test 검증 데이터 분리 20%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train_split, y_val = train_test_split(train,train, test_size=0.2, random_state=42)

###
# 훈련/평가 데이터 분할 (7.5:2.5)
#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(train, test, test_size=0.2, random_state=42)
 
# 모델 학습
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train_split) # 변수 대.소문자 유의하라 ~

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
# 모델 성능 평가, accuracy 평가 점수 계산 print(accuracy)
# ROC AUC 계산
pred_proba = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val['product_interest_new'], pred_proba)
print("ROC AUC Score:", round(roc_auc, 4))

# 정확도, 정밀도, 재현율 출력
# 레이블 예측
pred_label = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val['Target'], pred_label))
print("Precision:", precision_score(y_val['Target'], pred_label))
print("Recall:", recall_score(y_val['Target'], pred_label))
# 테스트 데이터 예측

# 테스트 데이터 결과 제출
submission_df = pd.DataFrame({'id': test.id, 'product_interest': pred_label})
submission_df.to_csv("final_submission.csv", index=False)

