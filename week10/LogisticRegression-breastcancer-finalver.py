# ♡ 데이터 준비 및 탐색
# ♡1. 데이터 준비하기 
import numpy as np
import pandas as pd

# 사이킷런에서 제공하는 데이터셋 중 유방암 진단 데이터셋 사용위해 load_breast_cancer
from sklearn.datasets import load_breast_cancer

# 데이터셋 로드하여 객체 b_cancer 사용 
b_cancer = load_breast_cancer()

# ♡2. 데이터 탐색하기
# 데이터셋에 대한 설명을 확인
print(b_cancer.DESCR)

# 데이터셋 객체 data 배열 b_cancer.data, 
# 즉 독립변수 X가 되는 피처를 DataFrame 자료형으로 변환하여 b_cancer_df를 생성
b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)

# 유방암 유무 class로 사용할 diagnosis 칼럼을 b_cancer_df에 추가하고 
# 데이터셋 객체의 target 칼럼 b_cancer.target 을 저장
b_cancer_df['diagnosis'] = b_cancer.target

# b_cancer_df의 데이터 샘플 5개를 출력 
b_cancer_df.head()

# ♡3. 데이터셋의 크기와 독립 변수 X가 되는 피처에 대한 정보를 확인 (오옹ㅎㅎ  shape은 크기 확인 ! ㅋㅋ)

# b_cancer_df.shape 사용해서 데이터셋의 행(샘플)의 개수, 열(변수)의 개수 확인
print('유방암 진단 데이터셋의 크기: ', b_cancer_df.shape)

# b_cancer_df 에 대한 정보 확인 (30개 피처,  )
b_cancer_df.info() # diagnosis는 악성이면1, 양성이면 0의 값이므로 유방암 여부에 대한 이진분류의 class로 사용할 종속변수가 됨

# ♡4. 로지스틱 회귀 분석에 피처로 사용할 데이터를 평균이 0, 분산이 1이 되는 정규 분포 형태로 맞춤

# 사이킷런의 전처리 패키지에 있는 정규 분포 스케일러를 임포트하고, 사용할 객체 scaler를 생성
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

# 피처로 사용할 데이터 b_cancer.data 에 대해 정규 분포 스케일링을 수행(scaler.fit)
b_cancer_scaled = scaler.fit_transform(b_cancer.data)

# 정규 분포 스케일링 후 값 조정됨 확인 
print(b_cancer_scaled[0])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ♤ 분석 모델 구축 및 결과 분석
# ♤1. 로지스틱 회귀를 이용하여 분석 모델 구축하기
# 필요한 모듈을 임포트
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X, Y 설정하기
Y = b_cancer_df['diagnosis'] # 종속변수 Y (0,1 범주형 변수)
X = b_cancer_scaled # 독립변수 X

# 훈련용 데이터와 평가용 데이터 분할하기
# 학습 데이터: 평가 데이터 = 7:3 으로 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)

# 로지스틱 회귀 분석: (1) 모델 생성 
# 로지스틱 회귀분석 모델 객체 lr_b_cancer 만듦
lr_b_cancer = LogisticRegression()

# 로지스틱 회귀 분석: (2) 모델 훈련
# 학습 데이터 X_train, Y_train 으로 모델 학습을 수행(fit)함
lr_b_cancer.fit(X_train, Y_train)

# 로지스틱 회귀 분석: (3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기 
# 학습이 끝난 모델에 대한 평가 데이터 X_test를 가지고 예측을 수행predict() 하여 예측값Y_predict 구함
Y_predict = lr_b_cancer.predict(X_test)


# ♤2. 생성한 모델의 성능 확인하기
# 필요한 모듈을 임포트
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 오차 행렬
# 평가를 위해 7:3으로 분할한 171개의 test 데이터에 대해 
# 이진 분류의 성능 평가 기본이 되는 오차 행렬을 구함
# 실행 결과를 보면 TN = 60개, FP=3개, FN=1개, TP=107개인 오차 행렬이 구해짐
confusion_matrix(Y_test, Y_predict)

# 성능 평가 지표인 정확도, 정밀도, 재현율, F1 스코어, ROC-AUC 스코어를 구함
accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict)
recall = recall_score(Y_test, Y_predict)
f1 = f1_score(Y_test, Y_predict)
roc_auc = roc_auc_score(Y_test, Y_predict)

# 성능 평가 지표를 출력하여 확인
print('정확도: {0:.3f}, 정밀도: {1:.3f}, 재현율: {2:.3f}, F1 : {3:.3f}'.format(accuracy, precision, recall, f1))

print('ROC_AUC: {0:.3f}'.format(roc_auc))




























