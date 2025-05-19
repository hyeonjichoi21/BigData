import numpy as np
import pandas as pd

# creating
df = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'])

df[1,1] # 안 됨
df[1][1] # 됨(인덱스, 콜론 값 아무것도 설정 안 하면)

df['중간고사'][1]  # 됨(콜론 값 설정했으니까 명시해주면)
df[1]['중간고사'] # 안 됨 (콜론 값 설정한 거니까)



df = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])

df['중간고사'][1] # 되긴 한데 안 좋다 (FutureWarning) 
df['중간고사']['2반']  # 됨 
df[1]['2반'] # 안 됨 



df['1반']['중간고사'] # 안 됨
type(df) # dataFrame 타입

# indexing-selection-assigning
df['중간고사']  # 열 뽑기 됨 (2차원처럼 보이지만, 1차원 !!)
type(df['중간고사']) # 타입이 Series 임 
df['중간고사'][0:2] # 0, 1의 행을 가져와라 
type(df['중간고사'][0]) # 숫자 행렬은 타입이 numpy
type(df['중간고사'][0:2]) # 타입이 Series (단순 값이 아니고 1차원 Series)
df['중간고사']['1반':'2반'] # 인덱스 값을 직접 지정할 수도 O
type(df['중간고사']['1반':'2반']) # 타입은 Series 

# loc
df.loc['1반'] # Object타입 (가로로 뽑아줌 ~ )
type(df.loc['1반']) # Series 
df.loc[:, '중간고사']  # : <- 모두 다 
type(df.loc[:, '중간고사']) # 타입 Series
df.loc[:, '중간고사']; df.loc[:, ['중간고사']] 
type(df.loc[:, ['중간고사']]) # 이건 DataFrame 타입;; 다시 보장장 ㅎㅎ
df.loc['1반':'2반']['중간고사'] 
df.loc['1반', '중간고사'] # loc 인덱스 됨~1 다시 보장장 ㅎㅎ
type(df.loc['1반', '중간고사'])
df.loc['1반'][0] # FutureWarning 

df.iloc[0] # 인덱스 이름 말고 숫자로 쓰고 싶다! -> iloc
type(df.iloc[0]) # 세로 말고 가로도 Series가 된다. 
df.iloc[0]['중간고사']
type(df.iloc[0]['중간고사']) # numpy (걍 값 1개면 numpy인듯..? 아닌감; 다시 보장장ㅎㅎ)


df.loc[df.성적 == 'B'] # 조건을 씀 
df.loc[df['성적'] == 'B']  # 이게 정석인데 문자열 공백 없으면 위도 ㄱㅊ 


df.성적 == 'B' # Boolean이 됨. True False 

df.loc[[True, False, False, False, True]] # 조건 쓴 거랑 일맥상통함 (5개 값 전부 다 줘야 함)

df.loc[(df.성적 == 'A') & (df.중간고사 >= 90)] # 조건 식 들어갈 수 O
df.loc[df.성적.isin(['B', 'C'])] # 조건부 업데이트 

## summary function and maps
df.describe() # (숫자 데이터만) 기술통계가 나옴 
df.중간고사.describe() # Series describe도 가능
df.head(1) # 첫번째 1개만 보고 싶음
df.중간고사.unique() # 중복 제외 서로 다른 값 
df.중간고사.mean() # 평균
df.중간고사.value_counts()
df.성적.value_counts()
df_mean = df.중간고사.mean()
df.중간고사.map(lambda p: p - df_mean) # 중간고사 값 - 평균

## grouping and sorting
df.groupby('중간고사').중간고사.count()
df.groupby('중간고사').중간고사.min()
df.groupby(['중간고사']).중간고사.agg([len, min, max])
df.sort_values(by='중간고사')
df.sort_values(by='중간고사', ascending=False)
df.sort_index(ascending=False)

# data types and missing values
df.dtypes # 타입 확인해보면 String대신 Object라고 함 
df.중간고사.dtypes
df.loc['6반']=[10, 10, np.nan]
df[pd.isnull(df.성적)] # null인 값 골라냄 

# renaming and combining
df.rename(columns={'성적': '등급'}) # 실행결과만 바뀌고, 원래 df는 안 바뀐 거 확인
df = df.rename(columns={'성적': '등급'}) # 이렇게 직접 넣어줘야 바뀜(치환 O)
df.rename_axis("반이름", axis='rows') # 원래 설정 안 하면 비어있는데, "반이름"으로 넣어줌 

df1 = pd.DataFrame([[89.2, 92.5, 'B'], 
                   [90.8,92.8, 'A'], 
                   [89.9, 95.2, 'A'],
                   [89.9, 85.2, 'C'],
                   [89.9, 90.2, 'B']], 
    columns = ['중간고사', '기말고사', '성적'], 
    index = ['1반', '2반', '3반', '4반', '5반'])

df0=pd.concat([df, df1]) # 붙이는 건 concat 

dir(pd) # pd.에 붙을 수 있는 메소드 다 보여줌 

dir(df) # df.에 ~ 

help(df.groupby) # 긴 설명 중 Examples 봄 (help로 함수를 알아가깅)

#df.to_csv('scores.csv') # 엑셀에 파일 열면 한글이 깨짐

df.to_csv('scores.csv', encoding='cp949', index = False) # 이러면 한글 안 깨짐

mydf = pd.read_csv('scores.csv', encoding='utf-8') 




