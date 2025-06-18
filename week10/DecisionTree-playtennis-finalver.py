# sklearn metrics: scikit-learn 패키지 중 모형평가에 사용되는 서브 패키지
# classification_report: 주요 분류 측정 항목을 보여주는 보고서 모듈 
# confusion_matrix: 분류의 정확성을 평가하기 위한 오차행렬 계산 모듈
from sklearn.metrics import classification_report, confusion_matrix

# sklearn.model_selection: scikit-learn 패키지 중 클래스를 나눌 때, 함수를 통해 train/test를 나눌 때 모델 검증에 사용되는 서브 패키지
# train_test_split: 배열 또는 행렬을 임의로 훈련 및 테스트 하위 집합으로 분할하는 모듈
from sklearn.model_selection import train_test_split

# sklearn.tree: scikit-learn 패키지 중 분류 및 회귀를 위한 의사결정 트리 기반 모델 패키지
# DecisionTreeClassifier: 의사결정 트리 분류 모듈
from sklearn.tree import DecisionTreeClassifier

# sklearn: 기계학습 관련 패키지 
# tree: 분류 및 회귀를 위한 의사결정 트리 기반 모듈 
from sklearn import tree

# IPython.display : IPython 내에 정보를 보여주는 도구용 공용 API
# Image : raw 데이터가 있는 PNG, JPEG 이미지 객체를 만드는 모듈
from IPython.display import Image

import pandas as pd # 데이터를 구조화된 형식으로 가공.분석 해주는 자료구조 패키지
import numpy as np # Numerical Python 줄임말. 고성능 계산, 데이터 분석 패키지
import pydotplus # 그래프 생성하는 graphviz의 Dot 언어를 파이썬으로 제공하는 모듈
import os # 운영체제와 상호작용하기 위한 기본적인 기능

tennis_data = pd.read_csv('playtennis.csv')
tennis_data

# 각 칼럼(Outlook, Temperature..을 문자열 -> 숫자(int)타입으로 변경)
# * 의사결정 트리 분류 모델에 train, test 데이터 값으로 사용하기 위한 전처리 과정
tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)

tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)

tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)

tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)

tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 10)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 11)

# 전처리된 tennis_data 변수 확인 
tennis_data

# np.array을 이용해 추출한 데이터를 배열 형태로 변환 후, 변수 X에 저장
X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
# 같은 방식으로 Playtennis를 변수 y에 저장
y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))

# 일반적으로 train / test 의 비율 = train(7.5) : test(2.5))
X_train, X_test, y_train, y_test = train_test_split(X, y)


# 로드(load)된 의사결정 트리 분류 모듈을 변수 dt_clf에 저장 
dt_clf = DecisionTreeClassifier()
# 의사결정 트리 분류 듈이 저장된 변수 dt_clf의 함수 fit()에 
# 변수 X_train, y_train을 입력해 의사결정 트리 분류 모델 생성, 생성한 모델을 다시 변수 dt_clf에 저장 
dt_clf = dt_clf.fit(X_train, y_train)

# 입력한 X_test에 대한 클래스 예측 값을 변수 dt_prediction에 저장 
dt_prediction = dt_clf.predict(X_test)

# 오차행렬을 계산하는 모듈 confusion_matrix에 변수 y_test와 dt_prediction을 입력 
print(confusion_matrix(y_test,dt_prediction))

# 분류 측정 항목을 보여주는 모듈인 classification_report()에 변수 y_test, dt_prediction입력
print(classification_report(y_test, dt_prediction))

# IPython 내에서 그래프 생성할 수 있는 인터페이스 경로 추가 설정하는 부분 
#os.environ['PATH'] += os.pathsep + 'C:\Program Files\Graphviz/bin/'

# 트리표현 함수에 입력되는 파라미터 중 하나인 feature_names에 값을 입력하기 위해
# 변수 tennis_data 각 컬럼을 list 형태로 -> 변수 feature_names 에 저장
feature_names = tennis_data.columns.tolist()
# 저장된 변수 feature_names를 슬라이싱(0:4)해 Outlook, Temperature, Humidity, Wind의 컬럼 추출 후 다시 변수 feature_names애 저장
feature_names = feature_names[0:4]

# 트리 표현 함수에 입력되는 파라미터 중 하나인 class_names에 값을 입력하기 위해
# target class 값인 'Play No'와 'Play Yes'를 배열형태로 변수 target_name에 저장
target_name = np.array(['Play No', 'Play Yes'])

# tree 패키지 중 의사결정 트리를 dot 형식으로 내보내는 함수인 
# export_graphviz()를 이용해, 변수 dt_dot_data에 저장
# df_clf : 의사결정 트리 분류기 
# out_file = 의사결정 트리를 파일/문자열로 반환 
# feature_names = 각 features의 이름(문자열)
# class_names = 각 대상 class 이름을 오름차순으로 정렬 
# filled = True일 경우 분류를 위한 다수 클래스, 회귀 값의 극한/다중 출력의 노드 순도 나타내기 위한 색칠
# rounded = True일 경우 둥근 모서리가 있는 노드 상자를 그리고, Times-Roman 대신 Helvetica 글꼴 사용
# special_characters = True일 경우 특수 문자 표시 
dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  special_characters = True)

# Pydotplus 모듈 중 Dot 형식의 데이터로 정의된 그래프를 로드하는 함수 
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)

# 변수 dt_graph에 대한 정보를 png파일로 생성하는 함수 
Image(dt_graph.create_png())





























