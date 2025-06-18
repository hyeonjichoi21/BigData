from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# loading the iris dataset
iris = load_iris()

# training data 설정
x_train = iris.data[:-30] # 데이터의 시작부터 끝에서 30번째 하나 전까지를 선택
y_train = iris.target[:-30]

# test data 설정
# iris.data : iris data에서 feature 데이터를 호출
x_test = iris.data[-30:] # test feature data
# iris.target : iris data에서 target 데이터를 호출  
y_test = iris.target[-30:] # test target data
# iris target 데이터에서 마지막 30번째부터 끝까지 선택

# Training data의 target 출력
print(y_train)

# Test data의 target 출력
print(y_test)
# Test data의 target은 전부 2(virginica)
# Traing data와 test data의 분리가 합리하지 않음 


# Random Forest 분류기 생성
# RandomForestClassifier 클래스를 import
from sklearn.ensemble import RandomForestClassifier
# Sklearn.ensemble모델은 분류,회귀, 이상 탐지를 위한 ensemble-based 방법을 포함

# 10개의 tree를 가진 random forest 생성 
# tree의 개수 Random Forest 분류 모듈 생성 
rfc = RandomForestClassifier(n_estimators=10)
rfc

# 입력 데이터 이용해 분류기 학습
rfc.fit(x_train, y_train)

# Test data를 입력해 target data를 예측
prediction = rfc.predict(x_test)

# 예측 결과 precision과 실제 test data의 target을 비교 
print(prediction == y_test)


# Random Forest 분류기 성능 평가 (1)
# Random forest 정확도 측정
# rfc.score() : RandomForestClassifier 클래스 안에 있는 분류 결과의 정확도(Accuracy)를 계산하는 함수 
rfc.score(x_test, y_test)
# x_test: 테스트 데이터 feature 값
# y_test: 테스트 데이터 target 값


# Random Forest 분류기 성능 평가 (2)
from sklearn.metrics import accuracy_score # 분류결과 accuracy 계산 
from sklearn.metrics import classification_report # 분류결과 precision, recall 계산 


print("Accuracy is : ", accuracy_score(prediction, y_test))
print("=========================================================")
print (classification_report(prediction, y_test))

# Training data와 test data를 잘 분리 하지 못한 이유로 분류 성능이 낮음 ...


# Random Forest 성능 제고 방법

# Training, Test 데이터 재 생성  ...Retry~~...
from sklearn.model_selection import train_test_split
x= iris.data # iris 데이터의 feature값
y = iris.target # iri 데이터의 target값 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print(y_test) # 앞에서 마지막 30개 데이터를 선택한 문제의.. 테스트 값^^ 
print(Y_test) # Train_test_split()를 이용해 생성한 테스트 데이터 target 값

# 데이터를 무작위로 혼합한 후, x의 80%를 X_train, x의 20%를 X_test
# y의 80%를 Y_train, y의 20%를 Y_test

clf = RandomForestClassifier(n_estimators=10) # Random Forest
clf.fit(X_train, Y_train)
prediction_1 = clf.predict(X_test) # 데이터 분류결과 예측 

# print (prediction_1 == Y_test)
print("Accuracy is : ", accuracy_score(prediction_1, Y_test)) 
print("============================================================")
print(classification_report(prediction_1, Y_test))

# 분류기의 성능 accuracy, precision, recall 모두 제고
# Training 데이터를 제대로 선택하는 것이 매우 중요함 



# Random Forest 분류기 성능 높이는 방법
# 1) Tree 개수 (n_estimators)를 변경 (트리 개수 적당히 확장하면 모듈 성능 높일 수 있음.) (너무 많아지면 오히려 성능 낮아짐)
# 2) max_features 의 값을 변경 


# Initialize the model 
clf_2 = RandomForestClassifier(n_estimators=200, # Number of trees
                               max_features=4, # Num features considered
                               oob_score=True) # Yse OOB scoring

clf_2.fit(X_train, Y_train)
prediction_2 = clf_2.predict(X_test)

print(prediction_2 == Y_test)
print("Accuracy is : ", accuracy_score(prediction_2, Y_test))
print("============================================================")
print(classification_report(prediction_2, Y_test))


# 각 feature의 중요도 확인
for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
    print(feature, imp)
# 확인결과 'petal(꽃잎) width'가 젤 중요도가 높음 !!

# 분류기를 생성할 때 RandomForestClassifier()의 파라미터 'oob_score=True' 선택 ! 
