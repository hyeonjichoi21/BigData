#!/usr/bin/env python
# coding: utf-8

# # 1. 데이터 탐색: 단변량
# In[1]:
import pandas as pd
data=pd.read_csv('Ex_CEOSalary.csv', encoding='utf-8')
# In[2]:
data.info()
# In[3]:
data.head()
# ## 1-1. 범주형 자료의 탐색
# In[4]:
data['industry'].value_counts()
# In[5]:
#data['industry'] = data['industry'].replace([1,2,3,4], ['Service', 'IT', 'Finance', 'Others'])
data['industry'].value_counts()
# In[6]:
#get_ipython().run_line_magic('matplotlib', 'inline')
data['industry'].value_counts().plot(kind="pie")
# In[7]:
data['industry'].value_counts().plot(kind="bar")
# ## 1-2. 연속형 자료의 탐색
# In[8]:
data.info()
# In[9]:
data.describe()
# In[10]:
data.skew()
# In[11]:
data.kurtosis()

# ### pandas 제공 기술통계 함수 
# 
# - count:  NA 값을 제외한 값의 수를 반환 
# - describe:  시리즈 혹은 데이터프레임의 각 열에 대한 기술 통계 
# 
# - min, max: 최소, 최대값 
# - argmin, argmax:  최소, 최대값을 갖고 있는 색인 위치 반환 
# - idxmin, idxmanx:  최소 최대값 갖고 있는 색인의 값 반환 
# - quantile:  0부터 1까지의 분위수 계산 
# - sum: 합 
# - mean: 평균 
# - median: 중위값 
# - mad: 평균값에서 절대 평균편차 
# - var: 표본 분산 
# - std: 표본 정규분산 
# - skew: 표본 비대칭도 
# - kurt: 표본 첨도 
# - cumsum: 누적 합 
# - cummin, cummax: 누적 최소값, 누적 최대값 
# - cumprod: 누적 곱 
# - diff: 1차 산술차 (시계열 데이터 사용시 유용) 
# - pct_change: 퍼센트 변화율 계산 
# - corr: 데이터프레임의 모든 변수 간 상관관계 계산하여 반환
# - cov: 데이터프레임의 모든 변수 간 공분산을 계산하여 반환

# In[12]:
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
# In[13]:
data['salary'].hist(bins=50, figsize=(20,15))
# In[14]:
data['sales'].hist(bins=50, figsize=(20,15))

# # 2. 데이터 탐색: 이변량
# In[15]:
data.corr()
# In[16]:
data.corr(method="pearson")
# In[17]:
data.corr(method="spearman")
# In[18]:
data.corr(method="kendall")
# In[19]:
import matplotlib.pyplot as plt
plt.scatter(data['sales'], data['salary'], alpha=0.5)
plt.show()
# In[20]:
plt.scatter(data['roe'], data['salary'], alpha=0.5)
plt.show()
# In[21]:
data.groupby('industry')[['salary']].describe()
# # 3. 이상치 처리
# In[22]:
data.boxplot(column='salary', return_type='both')
# In[23]:
data.boxplot(column='sales', return_type='both')
# In[24]:
data.boxplot(column='roe', return_type='both')

# ## 3-1. salary 변수 이상치 처리
# In[25]:
Q1_salary = data['salary'].quantile(q=0.25)
Q3_salary = data['salary'].quantile(q=0.75)
IQR_salary = Q3_salary-Q1_salary
IQR_salary
# In[26]:
data_IQR=data[(data['salary']<Q3_salary+IQR_salary*1.5)& (data['salary']>Q1_salary-IQR_salary*1.5)] 
# In[27]:
data_IQR['salary'].hist()
# In[28]:
data_IQR.hist(bins=50, figsize=(20,15))
# In[29]:
data_IQR.corr()
# In[30]:
import matplotlib.pyplot as plt
plt.scatter(data_IQR['sales'], data_IQR['salary'], alpha=0.5)
plt.show()
# In[31]:
plt.scatter(data_IQR['roe'], data_IQR['salary'], alpha=0.5)
plt.show()
# ## 3-2. sales 변수 이상치 처리
# In[32]:
Q1_sales = data['sales'].quantile(q=0.25)
Q3_sales = data['sales'].quantile(q=0.75)
IQR_sales = Q3_sales-Q1_sales
IQR_sales
# In[33]:
data_IQR=data[(data['sales']<Q3_sales+IQR_sales*1.5)& (data['sales']>Q1_sales-IQR_sales*1.5) &
              (data['salary']<Q3_salary+IQR_salary*1.5)& (data['salary']>Q1_salary-IQR_salary*1.5)]
# In[34]:
data_IQR['sales'].hist()
# In[35]:
data_IQR.hist(bins=50, figsize=(20,15))
# In[36]:
data_IQR.corr()

# # 4. 변수 변환
## 4-1. log 변환
# In[37]:
import numpy as np
data['log_salary']=np.log(data['salary'])
data['log_sales']=np.log(data['sales'])
data['log_roe']=np.log(data['roe'])
# In[38]:
data.head()
# In[39]:
data.hist(bins=50, figsize=(20,15))
# In[40]:
data.corr()

# ## 4-2. 제곱근 변환
# In[41]:
data['sqrt_salary']=np.sqrt(data['salary'])
data['sqrt_sales']=np.sqrt(data['sales'])
data['sqrt_roe']=np.sqrt(data['roe'])
# In[42]:
data.head()
# In[43]:
data.hist(bins=50, figsize=(20,15))
# In[44]:
data.corr()

# # 5. 결측치 처리
# In[45]:
import pandas as pd
data=pd.read_csv('Ex_Missing.csv')
data
# ## 5-1. 결측치 확인
# ### 가. 전체 및 변수별 결측 확인
# In[46]:
# isnull(): 결측이면 True, 결측이 아니면 False 값 반환
pd.isnull(data)
data.isnull()
# In[47]:
# notnull(): 결측이면 False, 결측이 아니면 True 값 반환
pd.notnull(data)
data.notnull()
# In[48]:
# 변수(컬럼)별로 결측값 개수 확인: df.isnull().sum()
data.isnull().sum()
# In[49]:
# 특정 변수(컬럼)의 결측값 개수 확인: df.isnull().sum()
data['salary'].isnull().sum()
# In[50]:
# 변수(컬럼)별로 결측 아닌 값의 개수 확인: df.notnull().sum()
data.notnull().sum()
# In[51]:
# 특정 변수(컬럼)의 결측 아닌 값의 개수 확인: df.notnull().sum()
data['salary'].notnull().sum()
# ### 나. 행별 결측 확인 및 저장
# In[52]:
# 행(row) 단위로 결측값 개수 구하기 : df.isnull().sum(1)
data.isnull().sum(1)
# In[53]:
# 행(row) 단위로 결측값 개수 구해서 새변수 생성하기
data['missing']=data.isnull().sum(1)
data
# In[54]:
# 행(row) 단위로 실측값 개수 구하기 : df.notnull().sum(1)
del data['missing']
data['valid']=data.notnull().sum(1)
data
# ## 5-2. 결측값 제거: dropna()
# - 결측값이 있는 행 제거: delete row with missing values
# - 결측값이 있는 열 제거: delete column with missing values
# - 결측값이 있는 특정 행 또는 열 제거: delete specific row or column with missing values
# ### 가. 결측값 있는 행(row/case) 제거
# In[55]:
data_del_row=data.dropna(axis=0)
data_del_row
# ### 나. 결측값 있는 열(column/variable) 제거
# In[56]:
data_del_col=data.dropna(axis=1)
data_del_col
# ### 다. 결측값 있는 특정 행/열 제거
# In[57]:
data[['salary']].dropna()
# In[58]:
data[['salary', 'sales', 'roe', 'industry']].dropna()
# In[60]:
data[['salary', 'sales', 'roe', 'industry']].dropna(axis=0)
# In[61]:
data[['salary', 'sales', 'roe', 'industry']].dropna(axis=1)
# ## 5-3. 결측값 대체
# - 결측값을 특정 값으로 대체: replace missing valeus with scalar value
# - 결측값을 변수별 평균으로 대체: filling missing values with mean value per columns
# - 결측값을 다른 변수의 값으로 대체: filling missing values with another columns' values
# - 결측값을 그룹 평균값으로 대체: fill missing values by Group means

# In[62]:
import pandas as pd
data=pd.read_csv('Ex_Missing.csv')
data
# ### 가. 특정값으로 대체: df.fillna(value/string)
# In[63]:
# 결측값을 0을 대체
data_0 = data.fillna(0)
data_0
# In[64]:
# 결측값을 'missing' 문자로 대체
data_missing = data.fillna('missing')
data_missing
# In[65]:
# 결측값을 앞 방향으로 채우기: df.fillna(method='ffill' or 'pad')
data_ffill=data.fillna(method='ffill')
data_ffill
# In[66]:
# 결측값을 앞 방향으로 채우기: df.fillna(method='ffill' or 'pad')
data_pad=data.fillna(method='pad')
data_pad
# In[67]:
# 결측값을 뒷 방향으로 채우기: df.fillna(method='bfill' or 'backfill')
data_bfill=data.fillna(method='bfill')
data_bfill
# In[68]:
# 결측값을 뒷 방향으로 채우기: df.fillna(method='bfill' or 'backfill')
data_backfill=data.fillna(method='backfill')
data_backfill
# ### 나. 평균 대체: 
# - df.fillna(df.mean())
# - df.where(pd.notnull(df), df.mean(), axis='columns')
# In[69]:
# 평균으로 대체
data_mean=data.fillna(data.mean())
data_mean
# In[70]:
# 중위수로 대체
data_median=data.fillna(data.median())
data_median
# In[71]:
# 최대/최소로 대체
data_max=data.fillna(data.max())
data_max
# In[72]:
# 다른 변수 평균으로 대체
# salary 변수의 평균값으로 모든 결측값 대체
data_other_mean=data.fillna(data.mean()['salary'])
data_other_mean
# ### 다. 다른 변수 값으로 대체
# In[73]:
# sales의 결측값을 salary 값으로 대체
import numpy as np
data2=data.copy()
data2['sales_new'] = np.where(pd.notnull(data2['sales']) == True, data2['sales'], data2['salary'])
data2
# ### 라. 집단 평균값으로 대체
# In[75]:
# 산업(industry)별 평균 확인
data.groupby('industry').mean()
# In[74]:
# lamda 함수 
fill_mean_func = lambda g: g.fillna(g.mean())
# In[75]:
# lamda 함수의 apply() 적용
data_group_mean=data.groupby('industry').apply(fill_mean_func)
data_group_mean
# - 집단별 특정 값으로 대체
# In[76]:
# 집단별로 변경할 값 설정
fill_values = {1: 1000, 2: 2000}
# In[77]:
# lamda 함수 적용
fill_func = lambda d: d.fillna(fill_values[d.name])
# In[78]:
# 집단별 apply
data_group_value=data.groupby('industry').apply(fill_func)
data_group_value
# - 변수별 다른 대체방법을 한번에 적용
# In[79]:
missing_fill_val = {'salary': data.salary.interpolate(),
                    'sales': data.sales.mean(),
                    'roe': 'missing'}
# In[80]:
print(missing_fill_val)
# In[81]:
data_multi = data.fillna(missing_fill_val)
data_multi

