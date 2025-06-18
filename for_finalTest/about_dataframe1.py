# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:32:13 2025

@author: chj
"""
"""Step1"""
import pandas as pd
data = {
        'apples': [3,2,0,1], 'oranges': [0,3,7,2]}

purchases = pd.DataFrame(data)

purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])


# Series 자료형 ~ ♡
col1 = pd.Series([3,2,0,1], name='apples')
col2 = pd.Series([3,2,0,1], name='oranges', index=['june', 'Robert', 'Lily', 'David'])

col1.name
col1.values


purchaes2= pd.DataFrame(col1)
purchases3= pd.DataFrame(col2)

col2= pd.Series([3,2,0,1], name='oranges')

# 같은 값이던 말든 그냥 중복으로 합쳐짐
purchases4=pd.concat([col1, col2], axis=1) 

"""Step2 - Series"""


data2 = ['1반', '2반', '3반', '4반', '5반']
sr4 = ['월', '화', '수', '목', '금']

sr8=pd.Series(data2, index = sr4)

sr8[2]

sr8['수']

sr8[-1]

"""Step3 - DataFrame"""
data_dic={
    'year': [2018, 2019, 2020],
    'sales' : [350, 480, 1099]
    }

df1 = pd.DataFrame(data_dic)

data2 = ['1반', '2반', '3반', '4반', '5반']
df2 = pd.DataFrame([[89.2, 92.5, 90.8], [92.8, 89.9, 95.2]],
                   index=['중간고사', '기말고사'],
                   columns=data2[0:3]
                   
                   )


df2.head(2)
df2.tail(1)
df2['1반'] # columns 이름으로 열검색 가능


"""Step4 - pandas indexing"""
df = pd.DataFrame([[60,61,62], [70,71,72], [80, 81, 82], [90,91,92]],
                  index=['1반', '2반', '3반', '4반'],
                  columns=['퀴즈1', '퀴즈2', '퀴즈3']
                  )


# df: 열 선택
df.퀴즈1
df['퀴즈1']
df['퀴즈1'][2] # 첫번째 열이 -> 인덱스 0 ~ ♡
df['2반':'3반'] # error -> df['2반', '3반']

df['퀴즈1'][3]
df.iloc[3,0]

# df.loc: 행 선택, 행열선택
df.loc['2반']
df.loc['2반', '퀴즈1']
df.loc['중간고사'] # -> 첫 매개변수에 열이름 넣으면 error !

df.loc['2반':'4반', '퀴즈1'] 
type(df.loc['2반':'4반', '퀴즈1'] ) # Series 타입 ~ ♡


# df.iloc: 행 선택, 행열선택
df.iloc[2]
df.iloc[2,1] # [행 인덱스, 열 인덱스] 2번째 행, 1번째 열

df.iloc[2:4, 0]
df.iloc[2:4, 0:2]


# df.loc vs df.iloc 차이점
# 1. 라벨(이름)기반 vs 정수(Integer) 기반
# 2. 끝 포함 vs 끝 미포함 
 

df[df.퀴즈3 == 62] # 해당하는 가로라인 출력 됨

df.loc[df.퀴즈3 == 62] # 이래도 해당하는 가로라인 출력 됨

# isin 함수
df[df.퀴즈3.isin([62,72 ])]

# & 교집합
df[(df.퀴즈3 == 62) & (df.퀴즈1 >=60)]
df.loc[(df.퀴즈3 == 62) & (df.퀴즈1 >=60)]

df.describe() # count, mean, std, min, 25%, 50%, 75%, max ..
df.퀴즈1.describe()
 
df.퀴즈1.unique()
df.퀴즈1.mean()
df.퀴즈1.value_counts() # 각 값이 몇번 나왔는지


# 이건 뭐임? ;;; -> 평균 구한 후, 평균보다 얼만큼 모자란지/충분한지
df_mean = df.퀴즈1.mean()
df.퀴즈1.map(lambda p: p -df_mean)

# grouping & sorting 
df.groupby('퀴즈1').퀴즈1.count()
df.groupby('퀴즈1').퀴즈1.min()
df.groupby('퀴즈1').퀴즈1.agg([len, min, max])

df.sort_values(by='퀴즈1')
df.sort_values(by='퀴즈1', ascending= False) # 내림차순
df.sort_index(ascending= False) # 인덱스 기준 내림차순

import numpy as np

# data types and missing values
df.dtypes
df.퀴즈1.dtypes
df.loc['5반'] = [50, 50, np.nan]
df[pd.isnull(df.퀴즈2)]


"""Step5 - pandas readCSV"""
df_score = pd.read_csv('C:\BigData\week3\scores.csv',encoding='cp949', index_col=0, engine='python')


"""Step6 - DafaFrame 자료 보기 명령어"""

# 1. pandas DataFrame 
len(df) # 행 개수 세기
df.shape[0] # 행 개수 세기
len(df.index)

df.shape[1] # 열 개수 세기
len(df.columns) # 열 개수 세기

df.count() # Null 값 아닌 행 개수 세기

df.groupby('퀴즈1').size() # 그룹별 행 개수 세기

df.groupby('퀴즈1').count()  # 그룹별 Null 값 아닌 행 개수 세기





# ++)  1. age가 30 이상이고, gender가 'M'인 행의 income 컬럼만 추출하는 코드



df.loc[(df['age'] >= 30) & (df['gender'] == 'M'), 'income']


# ++) 2. gender별 평균 income을 구하는 코드
df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'income': [3000, 2500, 4000, 2700, 3500]
})

df.groupby('gender').income.mean()


testdf=pd.DataFrame({'first': [1,2,3], 'second': [5,6,7]})
 
"""
[오차 행렬 구하는 식 정리]

정확도: TN+TP/전체 데이터 수
정밀도: TP/FP+TP (예측 P중에서 ~)
재현율, 민감도, TPR : TP/FN+TP (실제 P중에서 ~)
F1 스코어: 2 x (정밀도x재현율) / 정밀도 + 재현율 

p10에 잘 나와있음 <- 정 헷갈리면 봐라..타임어택 있으니까 

FPR : FP/FP+TN

특이도(specifity): TN/TN+FP




[CAP 이론]
첫번째 교안 
p11, 13, 14


[tf-idf]
chapter13 텍스트마이닝
p8




[강의안별 주요 개념]

chapter13 텍스트 마이닝:
  텍스트 마이닝, 특성 벡터화, BOW, 카운트 기반 벡터화, TF-IDF
  감성 분석(오피니언 마이닝), 토픽 모델링, LDA, pyLDAvis 


chapter12 군집 분석: 
  비지도 학습, 군집화, k-means, k-평균 알고리즘, 엘보 방법, 실루엣 분석
  

ch11 분류 분석 - Decision tree:
   Decision Tree, (분리기준)- 순수도, 불순도, 지니 지수, 엔트로피 지수, 정보 이득
   앙상블, voting, Bagging, Boosting, RandomForest, OOB, 



chapter 11 분류 분석 LogisticRegression:
   로지스틱 회귀, 시그모이드 함수, 오차행렬(혼돈행렬), 정확도, 정밀도, 재현율, F1 스코어,
   ROC 기반 AUC 스코어, 


chapter 10 회귀 분석:
   범주형 변수(명, 순), 수치형 변수(구, 비), 상관관계, 인과관계 
   인과관계 분석 - 회귀 분석, 단변량.다변량 단순.다중 선형 회귀 모델
   머신러닝 프로세스, 지도 학습, 분석 평가 지표 MAE, MSE, RMSE, R2
   

chapter9 지리 정보 분석: 
   오픈 웹 서비스 - geoService, 지도 정보 시각화 라이브러리 - 포리움, folium, 블록맵 

chapter8 텍스트 빈도 분석: 
   텍스트 분석, 워드 클라우드, 형태소, 품사 태깅

chapter 7 기초 통계 분석: 
   기술 통계, t-검정, 선형 회귀 분석, 히스토그램, 부분 회귀 플롯, 상관 계수 p, 피어슨상관계수, 산점도, 히트맵 

chapter 7 기초 통계 파이썬 실습: 
   변수와 척도, 도수분포표와 히스토그램, 연속변수, 중심경향치(평균, 최빈치, 중앙값), 변산성 측정치(분산, 표준편차, 범위, 사분위간 범위), 공분산, 상관계수, 상관계수의 통계적 검증, 상관분석 유의할 점, 회귀분석, 절편의 고정, 다중공산성, 잔차분석, t 검증, 독립표본 t 검증, 대응표본 t 검증, 
    



chapter6 - 웹페이지 크롤링:
   (정적 웹 페이지) BeautifulSoup, (동적 웹 페이지) Selenium, WebDriver

chapter6 - 오픈 API 크롤링:
   크롤링, 웹API

chapter6 - 데이터 제공 사이트


chapter5 - 대중교통 데이터 시각화 
   
chapter4 - 기온데이터 시각화
   csv 파일이란, matplotlib 라이브러리, 버블 차트 등 

chapter3 - 데이터 분석을 위한 주요 라이브러리 
   Series, Dataframe, 한글 코드, matplotlib

chapter2 - 파이썬 프로그래밍

chapter1-2 - 빅데이터 액셀 

chapter1 - 빅데이터 개요



"""
