from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#loading the iris dataset
iris = load_iris()

# to excel... by Uchang
import pandas as pd 
df = pd.DataFrame(data=iris['data'], columns = iris['feature_names'])
df.to_excel('iris.xlsx', index=False)

#training data 설정 
x_train = iris.data[:-30]
y_train = iris.target[:-30]
#test data 설정
x_test = iris.data[-30:] # test feature data  
y_test = iris.target[-30:] # test target data

#RandomForestClassifier libary를 import
from sklearn.ensemble import RandomForestClassifier # RandomForest
# RandomForest 는 ensemble 패키지에 들어 있음

#tree 의 개수 Random Forest 분류 모듈 생성
rfc = RandomForestClassifier(n_estimators=10) 
rfc
rfc.fit(x_train, y_train)
#Test data를 입력해 target data를 예측 
prediction = rfc.predict(x_test)
#예측 결과 precision과 실제 test data의 target 을 비교 
print (prediction==y_test)
#Random forest 정확도 츶정
rfc.score(x_test, y_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print ("Accuracy is : ",accuracy_score(prediction, y_test))
print ("=======================================================")
print (classification_report(prediction, y_test))
# 여기까지가 안 되는 경우 -> 데이터가 엉망인 케이스 

from sklearn.model_selection import train_test_split
x = iris.data
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2) # 이게 핵심 <- 이렇게 나누어야 random으로 고를 수 ㅒ
print (y_test) # 이 y test는 엉망
print (Y_test) # 이 Y test는 꽃이 섞여있어서 잘 됨
clf = RandomForestClassifier(n_estimators=10) # Random Forest
clf.fit(X_train, Y_train)
prediction_1 = rfc.predict(X_test)
#print (prediction_1 == Y_test)
print ("Accuracy is : ",accuracy_score(Y_test, prediction_1))
print ("=======================================================")
print (classification_report(prediction_1, Y_test))

# 자꾸 정확도 1.0 나옴;;;;;;;;;; <- 무조건 이상 있는 거다. 과제 시, 꼭 고쳐라 ★★★

# Initialize the model
clf_2 = RandomForestClassifier(n_estimators=200, # Number of trees
                               max_features=4,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*
clf_2.fit(X_train, Y_train)
prediction_2 = clf_2.predict(X_test)
print (prediction_2 == Y_test)
print ("Accuracy is : ",accuracy_score(prediction_2, Y_test))
print ("=======================================================")
print (classification_report(prediction_2, Y_test))

for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
    print(feature, imp)
    
import graphviz
import os
#os.environ['PATH'] += os.pathsep + 'c:\programdata\anaconda3\lib\site-packages'
estimator = clf_2.estimators_[5]
from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# 생성된 .dot 파일을 .png로 변환
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=50'])

# jupyter notebook에서 .png 직접 출력
from IPython.display import Image

# 다음 명령어는 따로 실행해본다.
Image(filename = 'decistion-tree.png')
