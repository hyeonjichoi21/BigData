#!/usr/bin/env python
# coding: utf-8
# # 12장. 군집분석 : 타깃마케팅을 위한 K-평균 군집화
# ### 1) 데이터 수집
import pandas as pd
import math

print('---- wait... 데이터 읽는중 takes 1 minute ....')
retail_df = pd.read_excel('./DATA/Online_Retail.xlsx')
retail_df.head()
print('---- end...')

# ### 2) 데이터 준비 및 탐색v
# 데이터 정보 확인하기 
retail_df.info()

# 오류 데이터 정제
retail_df = retail_df[retail_df['Quantity'] > 0] # Quantity가 0보다 작은 거 버림
retail_df = retail_df[retail_df['UnitPrice'] > 0] # 0보다 작은 거 버림
retail_df = retail_df[retail_df['CustomerID'].notnull()] # null값인 거 버림 

# 'CustomerID' 자료형을 정수형으로 변환(astype(int))
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)

retail_df.info()
print(retail_df.isnull().sum()) # isnull = null 개수 세는 거 
print(retail_df.shape)

# 중복 레코드 제거
retail_df.drop_duplicates(inplace=True) # drop_duplicates = 중복 제거하는 함수(1개만 남겨라)
print(retail_df.shape) #작업 확인용 출력

# #### - 제품 수, 거래건 수, 고객 수 탐색 (수량 확인하기)
pd.DataFrame([{'Product':len(retail_df['StockCode'].value_counts()),
              'Transaction':len(retail_df['InvoiceNo'].value_counts()),
              'Customer':len(retail_df['CustomerID'].value_counts())}], 
             columns = ['Product', 'Transaction', 'Customer'],
            index = ['counts'])

retail_df['Country'].value_counts()

# 주문금액 컬럼 추가
retail_df['SaleAmount'] = retail_df['UnitPrice'] * retail_df['Quantity']
retail_df.head() #작업 확인용 출력

# #### - 고객의 마지막 주문후 경과일(Elapsed Days), 주문횟수(Freq), 주문 총액(Total Amount) 구하기
aggregations = {    # aggregations = 집계 함수
    'InvoiceNo':'count', # 해당 그룹의 인보이스 개수 (판매 건수)
    'SaleAmount':'sum', # 해당 그룹의 총 매출액
    'InvoiceDate':'max' # 가장 마지막 구매 날짜 
}

# 각 고객의 정보 추출하기 위해, CustomerID를 기준으로 그룹 만든다.
# 새로운 데이터프레임 객체 customer_df 만든다.
customer_df = retail_df.groupby('CustomerID').agg(aggregations)
customer_df = customer_df.reset_index() # 새로운 인덱스 설정하기 

customer_df.head()  #작업 확인용 출력

# 컬럼이름 바꾸기
customer_df = customer_df.rename(columns = {'InvoiceNo':'Freq', 'InvoiceDate':'ElapsedDays'})
customer_df.head() #작업 확인용 출력

# #### - 마지막 구매후 경과일 계산하기
import datetime 
customer_df['ElapsedDays'] = datetime.datetime(2011,12,10) - customer_df['ElapsedDays']
customer_df.head() #작업 확인용 출력

# 마지막 구매 후 몇 일이 지났는지를 날짜수로 환산하여 ElapsedDays를 구함 (시,분,초는 하루로 +1 해줌)
customer_df['ElapsedDays'] = customer_df['ElapsedDays'].apply(lambda x: x.days+1)
customer_df.head() #작업 확인용 출력

# #### - 현재 데이터 값의 분포 확인하기

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq'], customer_df['SaleAmount'], customer_df['ElapsedDays']], sym='bo')
plt.xticks([1, 2, 3], ['Freq', 'SaleAmount','ElapsedDays' ])
plt.show()

# #### - 데이터 값의 왜곡(치우침)을 줄이기 위한 작업 : 로그 함수로 분포 조정
import numpy as np
customer_df['Freq_log'] = np.log1p(customer_df['Freq']) # 작은 값에 대하여 log(제곱근)값을 씌우면 
customer_df['SaleAmount_log'] = np.log1p(customer_df['SaleAmount']) # 큰 값에 대하여 log(제곱근)값을 씌우면
customer_df['ElapsedDays_log'] = np.log1p(customer_df['ElapsedDays'])
customer_df.head()  #작업 확인용 출력

# 조정된 데이터 분포를 다시 박스플롯으로 확인하기
fig, ax = plt.subplots()
ax.boxplot([customer_df['Freq_log'], customer_df['SaleAmount_log'],customer_df['ElapsedDays_log']], sym='bo')
plt.xticks([1, 2, 3], ['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'])
plt.show() # log 값을 취해서, 값들의 차이(왜곡)를 줄임 !!


# ### 3) 모델 구축 : K-평균 군집화 모델 
# k-평균 군집화 모델링을 위한 KMeans
from sklearn.cluster import KMeans
# 실루엣 계수 계산에 사용할 silhouette_score, silhouette_samples를 임포트
from sklearn.metrics import silhouette_score, silhouette_samples
# k-평균 모델에 사용할 값을 위해 
# Freq_log, SaleAmout_log, ElapsedDays_log 칼럼을 X_features에 저장
X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']].values

# 정규 분포로 다시 스케일링하기 # 편차가 크니까 다시 스케일링 ㅋㅋ. StandardScaler
from sklearn.preprocessing import StandardScaler
# X_features를 정규 분포로 스케일링 StandardScaler().fit_transform 하여 X_features_scaled에 저장
X_features_scaled = StandardScaler().fit_transform(X_features)

# ### - 최적의 k 찾기 (1) 엘보우 방법
# 엘보 방법으로 클러스터 개수 k 찾기 

# k-평균 모델을 생성하고 KMeans(), 훈련하는 fit() 작업을 클러스터의 개수인 1부터 10 까지 반복하면서
# 왜곡 값 inertia_을 리스트 distortions에 저장 append()
distortions = []
for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모델 생성
    kmeans_i.fit(X_features_scaled)   # 모델 훈련
    distortions.append(kmeans_i.inertia_) # inertia 구함 
    
# 클러스터 개수에 따른 왜곡 값의 변화를 그래프로 그려서 plot() 시각화
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# 클러스터의 개수 k를 3으로 설정하여 k-평균 모델을 다시 구축한 뒤, 모델에서 만든 클러스터 레이블을 확인 
kmeans = KMeans(n_clusters=3, random_state=0) # 모델 생성

# 모델 학습과 결과 예측(클러스터 레이블 생성)
Y_labels = kmeans.fit_predict(X_features_scaled) 
customer_df['ClusterLabel'] = Y_labels
customer_df.head()  #작업 확인용 출력

# ## 4) 결과 분석 및 시각화
# ### - 최적의 k 찾기 (2) 실루엣 계수에 따른 각 클러스터의 비중 시각화 함수 정의
from matplotlib import cm

# 실루엣 계수를 구하고, 각 클러스터의 비중을 가로 바 차트barh()로 시각화하기 위해 silhouetteViz 함수를 정의

def silhouetteViz(n_cluster, X_features): 
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)
    
    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster):
        c_silhouettes = silhouette_values[Y_labels == c]
        c_silhouettes.sort()
        y_ax_upper += len(c_silhouettes)
        color = cm.jet(float(c) / n_cluster)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)
    
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : '+ str(n_cluster)+'\n'               + 'Silhouette Score : '+ str(round(silhouette_avg,3)))
    plt.yticks(y_ticks, range(n_cluster))   
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

# ### - 클러스터 수에 따른 클러스터 데이터 분포의 시각화 함수 정의
# 클러스터의 데이터 분포를 확인하기 위해 스캐터 차트로 시각화 

# 클러스터에 대한 데이터의 분포를 스캐터 차트scatter()로 시각화하기 위해 cluster Scatter 함수를 정의
def clusterScatter(n_cluster, X_features): 
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) #클러스터의 색상 설정
        c_colors.append(c_color)
        #클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1],
                     marker='o', color=c_color, edgecolor='black', s=50, 
                     label='cluster '+ str(i))       
    
    #각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                    marker='^', color=c_colors[i], edgecolor='w', s=200)
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 클러스터 개수가 3, 4, 5, 6인 경우의 실루엣 계수와 각 클러스터의 비중, 그리고 데이터 분포를 시각화하여 비교 

silhouetteViz(3, X_features_scaled) #클러스터 3개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(4, X_features_scaled) #클러스터 4개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(5, X_features_scaled) #클러스터 5개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(6, X_features_scaled) #클러스터 6개인 경우의 실루엣 score 및 각 클러스터 비중 시각화

# 클러스터 분포를 이용하여 최적의 클러스터 수를 확인 
# clusterScatter 함수를 호출하여 클러스터의 데이터 분포(원으로 표시)와 클러스터의 중심점 위치(삼각형으로 표시)를 시각화
clusterScatter(3, X_features_scaled) #클러스터 3개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(4, X_features_scaled)  #클러스터 4개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(5, X_features_scaled)  #클러스터 5개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(6, X_features_scaled)  #클러스터 6개인 경우의 클러스터 데이터 분포 시각화

# silhouetteViz 함수를 호출한 결과에서 클러스터가 4개인 경우가 더 좋은 것으로 나타났으므로, 최적의 클러스터 개수 k를 4로 결정
# ### 결정된 k를 적용하여 최적의 K-mans 모델 완성
best_cluster = 4

kmeans = KMeans(n_clusters=best_cluster, random_state=0)
Y_labels = kmeans.fit_predict(X_features_scaled) # 최적의 k-평균 군집화 모델의 레이블 예측값 Y_labels을 구함

customer_df['ClusterLabel'] = Y_labels
customer_df.head()   #작업 확인용 출력

# #### - ClusterLabel이 추가된 데이터를 파일로 저장
customer_df.to_csv('./DATA/Online_Retail_Customer_Cluster.csv')



# ## << 클러스터 분석하기 >>
# ### 1) 각 클러스터의 고객수 
# 클러스터의 특징을 살펴보기 위해 먼저 ClusterLabel을 기준으로 그룹을 만듦
# ClusterLabel을 기준으로 CustomerID의 count값을 추출
customer_df.groupby('ClusterLabel')['CustomerID'].count()

# 고객 클러스터에서 총 구매 빈도와 총 구매 금액, 마지막 구매 이후 경과일 정보를 추출하고, 구매 1회당 평균 구매 
# ### 2) 각 클러스터의 특징
customer_cluster_df = customer_df.drop(['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'],axis=1, inplace=False)

# 주문 1회당 평균 구매금액 : SaleAmountAvg
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']
customer_cluster_df.head()

# 클러스터별 분석
customer_cluster_df.drop(['CustomerID'],axis=1, inplace=False).groupby('ClusterLabel').mean()

# => 고객 클러스터 1은 다른 클러스터보다 구매 횟수가 월등히 높지만 구매당 평균 금액은 두 번째로 높음
# 구매당 평균 금액은 고객 클러스터 3이 가장 높음 






































