import pandas as pd
data=pd.read_csv("mtcars.csv")
print(data)

dir(pd)
pd.read_csv.__doc__
print(pd.read_csv.__doc__)
print(pd.DataFrame.head.__doc__)
print(data.info())

# 필요없는 열 삭제 - 'Unnamed: 0' 자동차 이름
X=data.iloc[:, 1:]
X=X.dropna()
# 잘못된 값 바꾸기 '*3'->'3'
print(X['gear'].unique())
X['gear']=X['gear'].replace('*3', '3').replace('*5', 5)
print(X.info())

# 종속변수 am 
X['am']=X['am'].replace('manual',0).replace('auto', 1) 
Y=X['am']
X=X.drop(columns='am')

 
# 학습데이터/테스트데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.3, random_state=10)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("roc_auc_score", roc_auc_score(y_test, y_test_predicted)) # 0.8571428571428572
print(accuracy_score(y_test, y_test_predicted))
print(precision_score(y_test, y_test_predicted))
print(recall_score(y_test, y_test_predicted))

# RandomForest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("roc_auc_score", roc_auc_score(y_test, y_test_predicted)) # 0.8571428571428572
print(accuracy_score(y_test, y_test_predicted))
print(precision_score(y_test, y_test_predicted))
print(recall_score(y_test, y_test_predicted))

# LogisticRegression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000) # lbfgs failed to converge
model.fit(x_train, y_train)
y_test_predicted=model.predict(x_test)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("roc_auc_score", roc_auc_score(y_test, y_test_predicted)) # 0.9285714285714286
print(accuracy_score(y_test, y_test_predicted))
print(precision_score(y_test, y_test_predicted))
print(recall_score(y_test, y_test_predicted))

# 확률값 예측
print('****', model.predict_proba(x_test))

# 결과
print(y_test_predicted)
result=pd.DataFrame(y_test_predicted)
result.to_csv('result.csv', index=False)
      
# 기타 분류 알고리즘
# SVM, KNN, MLPClassifier



