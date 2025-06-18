#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
X_train = pd.read_csv("XX_train.csv")
Y_train = pd.read_csv("YY_train.csv")

# 여기서부터 (6/09 수업에서 코드 쳐주심 )
X_test = pd.read_csv("X_test.csv")
X_train.info() # 범주형 데이터 어쩌구


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder();
X_train['Warehouse_block'] = le.fit_transform(X_train['Warehouse_block'])
# 실행하면 X_train의 Warehouse_block의 값이 다 "숫자"로 바뀐 것을 알 수 있다.

X_test['Warehouse_block'] = le.fit_transform(X_test['Warehouse_block'])
# X_test에 대해서도 수행. (좌우 대칭)


X_train['Mode_of_Shipment'] = le.fit_transform(X_train['Mode_of_Shipment'])
X_test['Mode_of_Shipment'] = le.fit_transform(X_test['Mode_of_Shipment'])

X_train['Product_importance'] = le.fit_transform(X_train['Product_importance'])
X_test['Product_importance'] = le.fit_transform(X_test['Product_importance'])

X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_test['Gender'] = le.fit_transform(X_test['Gender'])


# train-test 검증 데이터 분리 20%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)



# 분석에 필요하지 않은 컬럼 제거 <- 딱히 필요없

# 라벨 인코딩 - 명목형 변수 <- 이건 아까 함

# minmaxscaling <- 이건 안 한다 침

# 모델 생성1(RandomForest)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train['Target']) # 변수 대.소문자 유의하라 ~




# # 모델 성능 평가
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
pred1=model.predict(X_val) # 이걸 주고 항렬값 구함? (배송 될 확률, 안 될 확률 2차원??)
acc = accuracy_score(y_val['Target'], pred1); # 정확도 
print(acc);



#제출
pred=model.predict(X_test)
submit = pd.DataFrame({'ID' : X_test['ID'], 'Predicted' : pred})
submit.to_csv('submission.csv', index=False)



































