# -*- coding: utf-8 -*-
"""
실루엣 계수 구하기 https://zephyrus1111.tistory.com/193
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

X = np.array([[2, 3], [1, 3], [2, 2.5], [8, 3], [10, 3], [9, 4]]) # silhouette_coef => 0.8822250218687797
X = np.array([[2, 3], [1, 3], [2, 2.5], [12, 3], [11, 3], [12, 4]]) # silhouette_coef => 0.913370116740671
X = np.array([[2, 3], [1, 3], [2, 2.5], [3, 3], [5, 3], [4, 4]]) # silhouette_coef => 0.5412807938829989
#X = np.array([[2, 3], [9, 3], [6, 3]])

#클러스터 값들 위치에 따라서, silhouette_score이 달라진다. 멀리 떨어져있을수록 값 커지고 14줄, 적당한 가격으로 떨어져 있을수록 값 작아짐 15줄
data=pd.DataFrame(X, columns=['x', 'y'])
data.info()
data.plot(kind="scatter", x="x",y="y",figsize=(5,5),color="red")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
centers=kmeans.cluster_centers_
print(centers)

# cluster center 그려보기
plt.scatter(X[:,0],X[:,1], marker='o')
plt.scatter(centers[:,0], centers[:,1], marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig = plt.figure(figsize=(7,7))
fig.set_facecolor('white')
for i, label in enumerate(labels):
    if label == 0:
        color = 'blue'
    elif label ==1:
        color = 'red'
    else:
        color = 'green'
    plt.scatter(X[i,0],X[i,1], color=color)    
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 실루엣 함수 
def get_silhouette_results(X, labels):
    def get_sum_distance(target_x, target_cluster):
        res = np.sum([np.linalg.norm(target_x-x) for x in target_cluster])
        return res
    
    '''
    각 데이터 포인트를 돌면서 a(i), b(i)를 계산
    그리고 s(i)를 계산한다.
    
    마지막으로 Silhouette(실루엣) Coefficient를 계산한다.
    '''
    uniq_labels = np.unique(labels)
    silhouette_val_list = []
    for i in range(len(labels)):
        target_data = X[i]
 
        ## calculate a(i)
        target_label = labels[i]
        target_cluster_data_idx = np.where(labels==target_label)[0]
        if len(target_cluster_data_idx) == 1:
            silhouette_val_list.append(0)
            continue
        else:
            target_cluster_data = X[target_cluster_data_idx]
            temp1 = get_sum_distance(target_data, target_cluster_data)
            a_i = temp1/(target_cluster_data.shape[0]-1)
 
        ## calculate b(i)
        b_i_list = []
        label_list = uniq_labels[np.unique(labels) != target_label]
        for ll in label_list:
            other_cluster_data_idx = np.where(labels==ll)[0]
            other_cluster_data = X[other_cluster_data_idx]
            temp2 = get_sum_distance(target_data, other_cluster_data)
            temp_b_i = temp2/other_cluster_data.shape[0]
            b_i_list.append(temp_b_i)
 
        b_i = min(b_i_list)
        s_i = (b_i-a_i)/max(a_i, b_i)
        silhouette_val_list.append(s_i)
 
    silhouette_coef_list = []
    for ul in uniq_labels:
        temp3 = np.mean([s for s, l in zip(silhouette_val_list, labels) if l == ul])
        silhouette_coef_list.append(temp3)
    
    silhouette_coef = max(silhouette_coef_list) # silhouette_coef = max(silhouette_coef_list)
    return (silhouette_coef, np.array(silhouette_val_list))

# 실루엣 함수 테스트
silhouette_coef, silhouette_val_list = get_silhouette_results(X, labels)
print(silhouette_coef)


# 실루엣 함수 그리기
import seaborn as sns 
## 각 클러스터별로 Silhouette(실루엣) 값을 정렬한다.
uniq_labels = np.unique(labels)
sorted_cluster_svl = []
rearr_labels = []
for ul in uniq_labels:
    labels_idx = np.where(labels==ul)[0]
    target_svl = silhouette_val_list[labels_idx]
    sorted_cluster_svl += sorted(target_svl)
    rearr_labels += [ul]*len(target_svl)
 
colors = sns.color_palette('hls', len(uniq_labels))
color_labels = [colors[i] for i in rearr_labels]
 
fig = plt.figure(figsize=(6, 10))
fig.set_facecolor('white')
plt.barh(range(len(sorted_cluster_svl)), sorted_cluster_svl, color=color_labels)
plt.ylabel('Data Index')
plt.xlabel('Silhouette Value')
plt.show()

# 실루엣 함수 호출해보기
from sklearn.metrics import silhouette_score 
s=silhouette_score(X, labels, metric='euclidean')
print('silhouette_score=>', s)

# 직접구현한 함수
#np.mean(silhouette_val_list)