#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# 데이터 불러오기
x_train = pd.read_csv('x_train.csv', encoding='euckr')
x_test = pd.read_csv('x_test.csv', encoding='euckr')
y_train = pd.read_csv('y_train.csv', encoding='euckr')

# 테스트 데이터의 cust_id 저장
x_test_cust_id = x_test['cust_id']

# 불필요한 컬럼 제거
x_train.drop(columns=['cust_id'], inplace=True)
x_test.drop(columns=['cust_id'], inplace=True)
y_train.drop(columns=['cust_id'], inplace=True)

# 결측치 처리
x_train['환불금액'] = x_train['환불금액'].fillna(0)
x_test['환불금액'] = x_test['환불금액'].fillna(0)

# 라벨 인코딩 - 명목형 변수
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train['주구매상품'] = encoder.fit_transform(x_train['주구매상품'])
x_test['주구매상품'] = encoder.transform(x_test['주구매상품'])
x_train['주구매지점'] = encoder.fit_transform(x_train['주구매지점'])
x_test['주구매지점'] = encoder.transform(x_test['주구매지점'])

# 파생변수 생성
x_train['환불금액_new'] = (x_train['환불금액'] > 0).astype(int)
x_test['환불금액_new'] = (x_test['환불금액'] > 0).astype(int)
x_train.drop(columns=['환불금액'], inplace=True)
x_test.drop(columns=['환불금액'], inplace=True)

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# 분석에 필요하지 않은 컬럼 제거
x_train.drop(columns=['최대구매액'], inplace=True)
x_test.drop(columns=['최대구매액'], inplace=True)

# train-test 검증 데이터 분리 20%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 모델 생성2(RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=200, max_features=4, oob_score=True, random_state=42)
model1.fit(X_train, y_train_split.values.ravel())  # .values.ravel() → 1D 배열로 변환

# 모델 성능 평가
from sklearn.metrics import roc_auc_score
pred1 = model1.predict(X_val)
print('RF', roc_auc_score(y_val, pred1))

# 최종 예측 및 제출
test_pred = model1.predict_proba(x_test)[:, 1]
submit = pd.DataFrame({'cust_id': x_test_cust_id, 'gender': test_pred})
submit.to_csv('result.csv', index=False)
